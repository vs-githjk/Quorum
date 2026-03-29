"""
voice/speak.py — ElevenLabs TTS speaker for Quorum.

Converts text to MP3 audio using ElevenLabs eleven_turbo_v2 and injects
it into the meeting via a callback. Uses an asyncio.Queue so Quorum never
speaks over itself — requests are serialised and processed one at a time.

If ElevenLabs fails, the error is logged and the bot continues — TTS failure
should never crash the transcription pipeline.
"""

import asyncio
import io
import logging
import os
import re
import time
import traceback
from typing import Callable

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings

load_dotenv(override=True)

logger = logging.getLogger(__name__)

_MODEL_ID = "eleven_turbo_v2"
_VOICE_SETTINGS = VoiceSettings(stability=0.5, similarity_boost=0.75)
_MAX_SENTENCES = 2


class QSpeaker:
    """
    Text-to-speech engine for Quorum using ElevenLabs.

    All speak() calls are queued and processed one at a time so Quorum
    never speaks over itself. The inject_callback receives raw MP3 bytes
    and is responsible for sending them to the meeting (via RecallClient).

    Usage:
        speaker = QSpeaker(api_key, voice_id, inject_callback)
        await speaker.start()           # start queue worker
        await speaker.speak("Hello")    # queue a TTS request
        await speaker.stop()            # drain queue and stop
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        inject_callback: Callable,
    ) -> None:
        """
        Initialise QSpeaker.

        Args:
            api_key:          ElevenLabs API key.
            voice_id:         ElevenLabs voice ID to use for all speech.
            inject_callback:  Async callable that receives MP3 bytes and
                              injects them into the meeting via Recall.ai.
        """
        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._inject = inject_callback

        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._running = False

        logger.info(
            "QSpeaker initialised — voice_id=%s model=%s",
            voice_id,
            _MODEL_ID,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the asyncio queue worker coroutine.

        Must be called before speak(). Safe to call multiple times —
        only one worker runs at a time.
        """
        if self._running:
            logger.warning("QSpeaker already running")
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("QSpeaker queue worker started")

    async def stop(self) -> None:
        """
        Drain the queue, process remaining items, then stop the worker.

        Waits for all queued speech to complete before returning so the
        bot doesn't cut off mid-sentence on shutdown.
        """
        self._running = False
        if self._worker_task is not None:
            await self._queue.join()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("QSpeaker stopped")

    async def speak(self, text: str) -> None:
        """
        Queue a TTS request. Returns immediately — audio plays asynchronously.

        Enforces a maximum of 2 sentences. If the text is longer it is
        truncated and a warning is logged — Quorum should never speak more
        than 2 sentences at once in a meeting.

        Args:
            text: The text Quorum should say aloud.
        """
        cleaned = self._enforce_sentence_limit(text)
        if cleaned != text:
            logger.warning(
                "speak(): text truncated to %d sentences. Original: %r",
                _MAX_SENTENCES,
                text[:80],
            )
        await self._queue.put(cleaned)
        logger.debug("Queued speech: %r (queue size=%d)", cleaned[:60], self._queue.qsize())

    async def generate_only(self, text: str) -> bytes:
        """
        Generate MP3 bytes for text without injecting into the meeting.

        Used for testing and pre-generation. Does NOT go through the queue.

        Args:
            text: Text to synthesise.

        Returns:
            Raw MP3 audio bytes.

        Raises:
            RuntimeError if ElevenLabs returns no audio.
        """
        return await self._generate(text)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _worker(self) -> None:
        """
        Asyncio queue worker — processes one TTS request at a time.

        Runs until self._running is False and the queue is empty.
        Errors are caught per-item so one failure doesn't stop the worker.
        """
        logger.debug("Queue worker running")
        while self._running or not self._queue.empty():
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                audio = await self._generate(text)
                await self._inject(audio)
            except Exception:
                logger.error(
                    "_worker: failed to generate/inject audio for %r:\n%s",
                    text[:60],
                    traceback.format_exc(),
                )
            finally:
                self._queue.task_done()

    async def _generate(self, text: str) -> bytes:
        """
        Call ElevenLabs and return MP3 bytes. Logs latency in milliseconds.

        Runs the blocking ElevenLabs SDK call in a thread executor so it
        doesn't block the event loop.

        Args:
            text: Text to synthesise.

        Returns:
            Raw MP3 bytes.

        Raises:
            RuntimeError if ElevenLabs returns empty audio.
        """
        start = time.time()
        logger.debug("Calling ElevenLabs for: %r", text[:60])

        def _call() -> bytes:
            """Blocking ElevenLabs call — runs in thread executor."""
            chunks = self._client.text_to_speech.convert(
                voice_id=self._voice_id,
                text=text,
                model_id=_MODEL_ID,
                voice_settings=_VOICE_SETTINGS,
                output_format="mp3_44100_128",
            )
            buf = io.BytesIO()
            for chunk in chunks:
                buf.write(chunk)
            return buf.getvalue()

        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(None, _call)

        latency_ms = int((time.time() - start) * 1000)
        logger.info(
            "ElevenLabs TTS ready — latency=%dms bytes=%d text=%r",
            latency_ms,
            len(audio_bytes),
            text[:60],
        )

        if not audio_bytes:
            raise RuntimeError("ElevenLabs returned empty audio")

        return audio_bytes

    @staticmethod
    def _enforce_sentence_limit(text: str) -> str:
        """
        Truncate text to at most MAX_SENTENCES sentences.

        Splits on sentence-ending punctuation (. ! ?) and rejoins up to
        the limit. Appends a period if the truncated text doesn't end with
        punctuation.

        Args:
            text: Input text, potentially many sentences.

        Returns:
            Text with at most _MAX_SENTENCES sentences.
        """
        # Split into sentences on . ! ? followed by space or end-of-string
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= _MAX_SENTENCES:
            return text.strip()

        truncated = " ".join(sentences[:_MAX_SENTENCES])
        if not truncated[-1] in ".!?":
            truncated += "."
        return truncated


# ── Standalone test ───────────────────────────────────────────────────────────

async def _run_test() -> None:
    """
    Test QSpeaker by generating a spoken phrase and saving it to MP3.

    Does NOT inject into a meeting — uses generate_only() so no bot or
    meeting session is needed.

    Saves output to test_output.mp3 and prints latency in milliseconds.

    Run with:
        python3 -m voice.speak
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

    if not api_key:
        print("[ERROR] ELEVENLABS_API_KEY not set in .env")
        return
    if not voice_id:
        print("[ERROR] ELEVENLABS_VOICE_ID not set in .env")
        return

    print("\n=== QSpeaker test ===\n")

    async def noop_inject(audio: bytes) -> None:
        """No-op inject callback for testing."""
        pass

    speaker = QSpeaker(
        api_key=api_key,
        voice_id=voice_id,
        inject_callback=noop_inject,
    )

    test_text = (
        "Hello, I am Quorum. "
        "I found a relevant PR for this discussion."
    )

    print(f"Generating TTS for: {test_text!r}\n")

    start = time.time()
    try:
        audio_bytes = await speaker.generate_only(test_text)
        latency_ms = int((time.time() - start) * 1000)
    except Exception:
        print(f"[FAIL] ElevenLabs call failed:\n{traceback.format_exc()}")
        return

    output_path = "test_output.mp3"
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    print(f"[PASS] Audio generated")
    print(f"       Latency  : {latency_ms}ms")
    print(f"       Size     : {len(audio_bytes):,} bytes")
    print(f"       Saved to : {output_path}")

    # Verify sentence truncation
    long_text = (
        "First sentence here. "
        "Second sentence here. "
        "Third sentence that should be cut off. "
        "Fourth sentence also cut."
    )
    truncated = QSpeaker._enforce_sentence_limit(long_text)
    sentences_out = len(re.split(r"(?<=[.!?])\s+", truncated.strip()))
    assert sentences_out <= _MAX_SENTENCES, f"Expected ≤{_MAX_SENTENCES} sentences, got {sentences_out}"
    print(f"\n[PASS] Sentence limit enforced: {truncated!r}")
    print()


def test() -> None:
    """Entry point for direct module execution."""
    asyncio.run(_run_test())


if __name__ == "__main__":
    test()
