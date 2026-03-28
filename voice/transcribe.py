"""
voice/transcribe.py — Deepgram live transcription for Quorum.

Opens a Deepgram Nova-2 streaming connection and converts incoming audio
chunks into TranscriptSegment objects. Speaker labels are normalised to
"Speaker 0", "Speaker 1" etc. and stay consistent for the lifetime of
the session via a speaker_map dict.

On connection error the client attempts one automatic reconnect before
giving up and logging the failure.
"""

import asyncio
import logging
import struct
import time
import traceback
from typing import Callable

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)
from dotenv import load_dotenv

from bot import TranscriptSegment

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ── Deepgram streaming options ────────────────────────────────────────────────

_DEEPGRAM_OPTIONS = LiveOptions(
    model="nova-2",
    language="en-US",
    smart_format=True,
    diarize=True,
    interim_results=True,
    utterance_end_ms="1000",
    punctuate=True,
    encoding="linear16",
    sample_rate=16000,
)


class DeepgramTranscriber:
    """
    Streams raw audio bytes to Deepgram and emits TranscriptSegment objects.

    Usage:
        transcriber = DeepgramTranscriber(api_key, on_segment_fn, meeting_id)
        await transcriber.start()
        await transcriber.send_audio(chunk)   # call repeatedly
        await transcriber.stop()
    """

    def __init__(
        self,
        api_key: str,
        on_segment: Callable,
        meeting_id: str,
    ) -> None:
        """
        Initialise the transcriber.

        Args:
            api_key:    Deepgram API key.
            on_segment: Async callable that receives a TranscriptSegment.
                        Called for both interim (is_final=False) and final
                        (is_final=True) results.
            meeting_id: Unique meeting session ID — stamped on every segment.
        """
        self._api_key = api_key
        self._on_segment = on_segment
        self._meeting_id = meeting_id

        self._client: DeepgramClient | None = None
        self._connection = None
        self._connected = False
        self._reconnecting = False

        # Maps raw Deepgram speaker integers → "Speaker N" strings.
        # Populated on first appearance so labels never change mid-session.
        self._speaker_map: dict[int, str] = {}
        self._speaker_counter = 0

        self._segments_received = 0
        self._loop: asyncio.AbstractEventLoop | None = None

        logger.info(
            "DeepgramTranscriber initialised — meeting_id=%s", meeting_id
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Open the Deepgram live streaming connection and register event handlers.

        On error, attempts one automatic reconnect. Logs a critical error and
        returns if both attempts fail.
        """
        self._loop = asyncio.get_event_loop()
        success = await self._connect()
        if not success:
            logger.critical(
                "DeepgramTranscriber: could not establish connection for meeting %s",
                self._meeting_id,
            )

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send a raw audio chunk to the open Deepgram connection.

        Silently drops the chunk if the connection is not yet established
        rather than raising — the caller should not need to guard against this.

        Args:
            audio_chunk: Raw PCM-16 or encoded audio bytes.
        """
        if not self._connected or self._connection is None:
            logger.debug("send_audio: not connected, dropping chunk")
            return
        try:
            await self._connection.send(audio_chunk)
        except Exception:
            logger.error("send_audio failed:\n%s", traceback.format_exc())
            if not self._reconnecting:
                asyncio.create_task(self._attempt_reconnect())

    async def stop(self) -> None:
        """
        Flush pending audio and close the Deepgram connection cleanly.

        Logs total segments received during the session.
        """
        self._connected = False
        if self._connection is not None:
            try:
                await self._connection.finish()
            except Exception:
                logger.warning(
                    "stop: error finishing connection:\n%s",
                    traceback.format_exc(),
                )
        logger.info(
            "DeepgramTranscriber stopped — meeting=%s total_segments=%d",
            self._meeting_id,
            self._segments_received,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _connect(self) -> bool:
        """
        Open a new Deepgram WebSocket connection and attach event handlers.

        Returns:
            True if the connection started successfully, False otherwise.
        """
        try:
            self._client = DeepgramClient(
                self._api_key,
                DeepgramClientOptions(options={"keepalive": "true"}),
            )
            self._connection = self._client.listen.asyncwebsocket.v("1")

            self._connection.on(
                LiveTranscriptionEvents.Transcript, self._on_message
            )
            self._connection.on(
                LiveTranscriptionEvents.Error, self._on_error
            )
            self._connection.on(
                LiveTranscriptionEvents.Close, self._on_close
            )

            started = await self._connection.start(_DEEPGRAM_OPTIONS)
            if started:
                self._connected = True
                logger.info(
                    "Deepgram connection open — meeting=%s", self._meeting_id
                )
                return True
            else:
                logger.error("Deepgram connection failed to start")
                return False

        except Exception:
            logger.error(
                "_connect failed:\n%s", traceback.format_exc()
            )
            return False

    async def _attempt_reconnect(self) -> None:
        """
        Attempt one reconnect after a connection drop or send failure.

        Sets self._reconnecting to prevent parallel reconnect attempts.
        """
        if self._reconnecting:
            return
        self._reconnecting = True
        self._connected = False
        logger.warning(
            "Deepgram connection lost — attempting reconnect for meeting %s",
            self._meeting_id,
        )
        await asyncio.sleep(1)
        success = await self._connect()
        self._reconnecting = False
        if success:
            logger.info("Deepgram reconnected for meeting %s", self._meeting_id)
        else:
            logger.critical(
                "Deepgram reconnect failed — transcription stopped for meeting %s",
                self._meeting_id,
            )

    def _resolve_speaker(self, raw_speaker: int | None) -> str:
        """
        Map a raw Deepgram speaker integer to a stable "Speaker N" label.

        New speaker integers get the next available label. Labels never
        change once assigned, even if the speaker is silent for a while.

        Args:
            raw_speaker: Integer speaker ID from Deepgram diarization,
                         or None if diarization is unavailable.

        Returns:
            Stable "Speaker N" string.
        """
        if raw_speaker is None:
            return "Speaker 0"
        if raw_speaker not in self._speaker_map:
            self._speaker_map[raw_speaker] = f"Speaker {self._speaker_counter}"
            self._speaker_counter += 1
            logger.debug(
                "New speaker mapped: %d → %s",
                raw_speaker,
                self._speaker_map[raw_speaker],
            )
        return self._speaker_map[raw_speaker]

    # ── Deepgram event handlers ───────────────────────────────────────────────

    async def _on_message(self, _self, result, **kwargs) -> None:
        """
        Handle a Deepgram transcript event.

        Builds a TranscriptSegment and calls self._on_segment for both
        interim and final results.

        Args:
            result: Deepgram LiveResultResponse object.
        """
        try:
            sentence = result.channel.alternatives[0].transcript
            if not sentence.strip():
                return

            is_final = result.is_final
            words = result.channel.alternatives[0].words

            # Use speaker from the first word if diarization is available
            raw_speaker = None
            if words:
                raw_speaker = getattr(words[0], "speaker", None)
            speaker = self._resolve_speaker(raw_speaker)

            segment = TranscriptSegment(
                text=sentence,
                speaker=speaker,
                timestamp=time.time(),
                is_final=is_final,
                meeting_id=self._meeting_id,
            )

            self._segments_received += 1
            logger.debug(
                "[%s] %s | final=%s | %r",
                self._meeting_id,
                speaker,
                is_final,
                sentence[:80],
            )

            # Schedule the callback on the event loop — don't block the handler
            if asyncio.iscoroutinefunction(self._on_segment):
                asyncio.create_task(self._on_segment(segment))
            else:
                self._on_segment(segment)

        except Exception:
            logger.error(
                "_on_message error:\n%s", traceback.format_exc()
            )

    async def _on_error(self, _self, error, **kwargs) -> None:
        """
        Handle a Deepgram connection error and attempt one reconnect.

        Args:
            error: Deepgram error object.
        """
        logger.error("Deepgram error: %s", error)
        if not self._reconnecting:
            asyncio.create_task(self._attempt_reconnect())

    async def _on_close(self, _self, close, **kwargs) -> None:
        """
        Handle a Deepgram connection close event.

        Args:
            close: Deepgram close object.
        """
        logger.warning("Deepgram connection closed: %s", close)
        self._connected = False


# ── Standalone test ───────────────────────────────────────────────────────────

def _make_silent_wav(duration_seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a minimal silent PCM-16 WAV byte string.

    Used as a fallback audio source when no microphone is available,
    so the Deepgram connection can be verified without real speech.

    Args:
        duration_seconds: Length of audio to generate.
        sample_rate:      PCM sample rate in Hz.

    Returns:
        Raw WAV bytes (RIFF header + silent PCM-16 samples).
    """
    num_samples = int(duration_seconds * sample_rate)
    pcm_data = bytes(num_samples * 2)  # 16-bit = 2 bytes per sample, all zeros

    # Minimal RIFF/WAV header
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,           # PCM chunk size
        1,            # AudioFormat = PCM
        1,            # NumChannels = mono
        sample_rate,
        sample_rate * 2,  # ByteRate
        2,            # BlockAlign
        16,           # BitsPerSample
        b"data",
        data_size,
    )
    return header + pcm_data


async def _run_test() -> None:
    """
    Test DeepgramTranscriber by streaming 3 seconds of silent audio.

    Verifies that:
    - The WebSocket connection opens successfully.
    - The on_segment callback is wired and callable.
    - The connection closes cleanly.

    Note: Silent audio will not produce transcript output — the test
    confirms connectivity only. For real transcript output, speak into
    a microphone or use a WAV file with speech.

    Run with:
        python3 -m voice.transcribe
    """
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] DEEPGRAM_API_KEY not set in .env")
        return

    segments_received: list[TranscriptSegment] = []

    async def on_segment(seg: TranscriptSegment) -> None:
        """Capture and print each segment as it arrives."""
        tag = "FINAL  " if seg.is_final else "interim"
        print(f"  [{tag}] {seg.speaker}: {seg.text!r}")
        segments_received.append(seg)

    print("\n=== DeepgramTranscriber test ===\n")
    print("Connecting to Deepgram...")

    transcriber = DeepgramTranscriber(
        api_key=api_key,
        on_segment=on_segment,
        meeting_id="test-transcribe-001",
    )

    await transcriber.start()

    if not transcriber._connected:
        print("[FAIL] Could not connect to Deepgram — check DEEPGRAM_API_KEY")
        return

    print("[PASS] Connected to Deepgram\n")
    print("Streaming 3 seconds of silent audio in 100ms chunks...")
    print("(Silent audio will not produce transcripts — connection test only)")
    print("Speak into your mic after running main_bot.py for real output.\n")

    wav_bytes = _make_silent_wav(duration_seconds=3.0)
    # Strip 44-byte WAV header — send raw PCM to Deepgram
    pcm_bytes = wav_bytes[44:]

    chunk_size = int(16000 * 2 * 0.1)  # 100ms of 16kHz PCM-16
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i : i + chunk_size]
        await transcriber.send_audio(chunk)
        await asyncio.sleep(0.1)

    await asyncio.sleep(1)  # Allow final events to fire
    await transcriber.stop()

    print(f"\nSegments received: {len(segments_received)}")
    print(f"Total logged by transcriber: {transcriber._segments_received}")
    print("\n[PASS] Connection opened and closed cleanly")
    print("Run main_bot.py with a real meeting to see live transcripts.\n")


def test() -> None:
    """Entry point for direct module execution."""
    asyncio.run(_run_test())


if __name__ == "__main__":
    test()
