"""
main_bot.py — Quorum entry point.

Wires RecallClient, DeepgramTranscriber, QuorumSpeaker, and AudioStreamManager
into a single bot that joins a meeting, listens, and speaks back.

Run with:
    python3 main_bot.py
"""

import asyncio
import logging
import os
import uuid

import uvicorn
from dotenv import load_dotenv

load_dotenv(override=True)

from bot import BotStatus, TranscriptSegment
from bot.audio_stream import AudioStreamManager, get_app
from bot.recall_client import RecallClient
from voice.speak import QuorumSpeaker
from voice.transcribe import DeepgramTranscriber

# Agent brain
from agent import QuorumOrchestrator, SpeakCommand, ContextRequest, ActionRequest, IntegrationResult

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Env vars ──────────────────────────────────────────────────────────────────

_RECALL_API_KEY     = os.getenv("RECALL_API_KEY", "").strip()
_DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY", "").strip()
_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
_ELEVENLABS_VOICE   = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
_WEBHOOK_BASE_URL   = os.getenv("WEBHOOK_BASE_URL", "").strip()
_BOT_NAME           = os.getenv("BOT_NAME", "Quorum").strip()
_SERVER_PORT        = int(os.getenv("SERVER_PORT", "8000"))


class QuorumBot:
    """
    Top-level orchestrator that wires all Quorum subsystems together.

    Responsibilities:
    - Join the meeting via Recall.ai
    - Receive transcript segments from Recall.ai's WebSocket
    - Dispatch segments to the agent (AGENT HOOK)
    - Speak responses back via ElevenLabs + Recall.ai audio injection
    """

    def __init__(self) -> None:
        """
        Initialise all subsystems. No network calls made here —
        everything connects in start().
        """
        self._manager = AudioStreamManager()

        self._recall = RecallClient(
            api_key=_RECALL_API_KEY,
            webhook_base_url=_WEBHOOK_BASE_URL,
            bot_name=_BOT_NAME,
        )

        self._speaker = QuorumSpeaker(
            api_key=_ELEVENLABS_API_KEY,
            voice_id=_ELEVENLABS_VOICE,
            inject_callback=self._inject_audio,
        )

        # Transcriber is created per-meeting in start() so meeting_id is set
        self._transcriber: DeepgramTranscriber | None = None

        self._bot_status: BotStatus | None = None
        self._meeting_id: str | None = None
        self._server_task: asyncio.Task | None = None

        # ── Agent orchestrator ────────────────────────────────────────────
        self._orchestrator = QuorumOrchestrator(
            speak_callback=self._speak_command,
            integration_callback=self._integration_stub,
            action_callback=self._action_stub,
        )

        # Wire the segment callback into the manager
        self._manager.register_segment_callback(self.on_transcript_segment)

        logger.info("QuorumBot initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self, meeting_url: str) -> None:
        """
        Join a meeting and start all subsystems.

        Steps:
        1. Generate a unique meeting_id
        2. Join via Recall.ai
        3. Start FastAPI/uvicorn server in background
        4. Start ElevenLabs speaker queue
        5. Initialise meeting context in AudioStreamManager
        6. Block until KeyboardInterrupt

        Args:
            meeting_url: Full Google Meet / Zoom / Teams URL to join.
        """
        self._meeting_id = str(uuid.uuid4())[:8]  # short ID for readability

        logger.info("Quorum is starting up...")
        logger.info("Meeting ID : %s", self._meeting_id)
        logger.info("Health URL : %s/health", _WEBHOOK_BASE_URL)

        # ── Step 1: Start FastAPI server ──────────────────────────────────
        app = get_app(self._manager)
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=_SERVER_PORT,
            log_level="warning",    # suppress uvicorn noise
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.5)    # give uvicorn a moment to bind
        logger.info("Server listening on port %d", _SERVER_PORT)

        # ── Step 2: Start speaker queue ───────────────────────────────────
        await self._speaker.start()
        logger.info("Speaker ready")

        # ── Step 3: Join meeting via Recall.ai ────────────────────────────
        logger.info("Joining meeting: %s", meeting_url)
        self._bot_status = await self._recall.join_meeting(
            meeting_url, self._meeting_id
        )

        if self._bot_status.status == "error":
            logger.critical("Failed to join meeting — check RECALL_API_KEY and URL")
            await self.stop()
            return

        logger.info("Bot joining — bot_id=%s", self._bot_status.bot_id)

        # ── Step 4: Register session with manager + agent ────────────────
        self._manager.start_session(self._meeting_id, self._bot_status.bot_id)
        await self._orchestrator.start_meeting(self._meeting_id)

        # ── Step 5: Keep running until interrupted ────────────────────────
        try:
            await asyncio.Event().wait()    # block forever
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """
        Leave the meeting and shut down all subsystems cleanly.
        """
        logger.info("Quorum is leaving the meeting...")

        if self._bot_status and self._bot_status.bot_id:
            await self._recall.leave_meeting(self._bot_status.bot_id)

        if self._transcriber is not None:
            await self._transcriber.stop()

        await self._speaker.stop()

        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        if self._meeting_id:
            await self._orchestrator.end_meeting(self._meeting_id)
        self._manager.end_session()
        logger.info("Quorum has left the meeting")

    # ── Transcript handler ────────────────────────────────────────────────────

    async def on_transcript_segment(self, segment: TranscriptSegment) -> None:
        """
        Handle every incoming transcript segment from Recall.ai.

        Only processes final segments (is_final=True). Logs each segment
        in the format [Speaker 0] spoken text.

        For the MVP, responds to any segment containing "quorum" with a
        hardcoded reply to confirm the audio pipeline is working end-to-end.

        Args:
            segment: A TranscriptSegment from the WebSocket handler.
        """
        if not segment.is_final:
            return

        logger.info("[%s] %s", segment.speaker, segment.text)

        # ── AGENT HOOK — dispatch to orchestrator ─────────────────────────
        await self._orchestrator.process_segment(segment)
        # ─────────────────────────────────────────────────────────────────

    # ── Agent callbacks ───────────────────────────────────────────────────────

    async def _speak_command(self, cmd: SpeakCommand) -> None:
        """Receive a SpeakCommand from the orchestrator and speak it aloud."""
        logger.info("Speaking: %r", cmd.text[:80])
        await self._speaker.speak(cmd.text)

    async def _integration_stub(self, req: ContextRequest) -> list[IntegrationResult]:
        """
        Stub integration callback — replace with real integrations module.
        Returns empty list until integrations branch is merged in.
        """
        logger.info("Integration request (stub): query=%r sources=%s", req.query, req.sources)
        return []

    async def _action_stub(self, req: ActionRequest) -> dict:
        """
        Stub action callback — replace with real actions-ui module.
        Logs the action until actions-ui branch is merged in.
        """
        logger.info("Action request (stub): type=%s params=%s", req.action_type, req.parameters)
        return {"status": "stub", "action_type": req.action_type}

    # ── Audio injection ───────────────────────────────────────────────────────

    async def _inject_audio(self, audio_bytes: bytes) -> None:
        """
        Inject MP3 audio bytes into the meeting via Recall.ai.

        This is the inject_callback passed to QuorumSpeaker. Called
        automatically after ElevenLabs generates audio.

        Args:
            audio_bytes: Raw MP3 bytes from ElevenLabs.
        """
        if self._bot_status is None or not self._bot_status.bot_id:
            logger.warning("_inject_audio: no active bot — dropping audio")
            return
        await self._recall.inject_audio(self._bot_status.bot_id, audio_bytes)


# ── Entry point ───────────────────────────────────────────────────────────────

_ASCII_HEADER = """
=================================
       Q U O R U M
  AI Meeting Participant
=================================
"""


async def _main() -> None:
    """Async entry point — prompt for URL and start the bot."""
    bot = QuorumBot()
    url = input("Paste your meeting URL: ").strip()
    if not url:
        print("[ERROR] No URL provided — exiting")
        return
    try:
        await bot.start(url)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    print(_ASCII_HEADER)

    # Validate required env vars before starting
    missing = [
        name for name, val in {
            "RECALL_API_KEY":     _RECALL_API_KEY,
            "DEEPGRAM_API_KEY":   _DEEPGRAM_API_KEY,
            "ELEVENLABS_API_KEY": _ELEVENLABS_API_KEY,
            "ELEVENLABS_VOICE_ID":_ELEVENLABS_VOICE,
            "WEBHOOK_BASE_URL":   _WEBHOOK_BASE_URL,
        }.items() if not val
    ]
    if missing:
        print(f"[ERROR] Missing required env vars: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in all values.")
        raise SystemExit(1)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nShutting down...")
