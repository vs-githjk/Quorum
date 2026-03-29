"""
main_bot.py — Quorum entry point.

Wires RecallClient, DeepgramTranscriber, QSpeaker, and AudioStreamManager
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
from voice.speak import QSpeaker
from voice.transcribe import DeepgramTranscriber

# Agent brain
from agent import QOrchestrator, QAgent, SpeakCommand, ContextRequest, ActionRequest, IntegrationResult

# Integrations
from integrations import integration_callback
from integrations.github import search_github
from integrations.notion import search_notion
from integrations.slack import search_slack
from integrations.asana import search_asana, create_asana_task, update_asana_task

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
_BOT_NAME           = os.getenv("BOT_NAME", "Q").strip()
_SERVER_PORT        = int(os.getenv("SERVER_PORT", "8000"))
_SCREEN_API_URL     = os.getenv("SCREEN_API_URL", "http://localhost:5000").rstrip("/")
_QUORUM_MODE        = os.getenv("QUORUM_MODE", "on_demand").strip()
_OPENROUTER_KEY     = os.getenv("OPENROUTER_API_KEY", "").strip()
_OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
_OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

_CANCEL_WORDS = frozenset({"stop", "cancel", "abort", "halt"})


def _fmt_results(results: list) -> str:
    """Format a list[IntegrationResult] into a readable string for the LLM."""
    if not results:
        return "No results found."
    lines = []
    for r in results:
        line = f"- [{r.source}] {r.title}: {r.summary}"
        if r.url:
            line += f" | url: {r.url}"
        # Expose task GID so the LLM can call update_asana_task
        if r.source == "asana" and isinstance(r.raw_data, dict) and r.raw_data.get("gid"):
            line += f" | task_gid: {r.raw_data['gid']}"
        lines.append(line)
    return "\n".join(lines)


class QBot:
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

        self._speaker = QSpeaker(
            api_key=_ELEVENLABS_API_KEY,
            voice_id=_ELEVENLABS_VOICE,
            inject_callback=self._inject_audio,
        )

        # Transcriber is created per-meeting in start() so meeting_id is set
        self._transcriber: DeepgramTranscriber | None = None

        self._bot_status: BotStatus | None = None
        self._meeting_id: str | None = None
        self._server_task: asyncio.Task | None = None
        self._screen_action_running: bool = False
        self._screen_link_sent: set[str] = set()

        # ── Agent orchestrator ────────────────────────────────────────────
        self._orchestrator = QOrchestrator(
            speak_callback=self._speak_command,
            integration_callback=self._integration_stub,
            action_callback=self._action_stub,
            chat_callback=self._send_chat,
        )

        # ── QAgent with tool registrations ───────────────────────────────
        context = self._orchestrator._context
        agent = QAgent(context=context)

        async def _tool_search_slack(meeting_id: str, query: str) -> str:
            return _fmt_results(await search_slack(query))

        async def _tool_search_notion(meeting_id: str, query: str) -> str:
            return _fmt_results(await search_notion(query))

        async def _tool_search_github(meeting_id: str, query: str) -> str:
            return _fmt_results(await search_github(query))

        async def _tool_search_asana(meeting_id: str, query: str) -> str:
            return _fmt_results(await search_asana(query))

        async def _tool_create_asana_task(meeting_id: str, title: str, notes: str = "") -> str:
            return await create_asana_task(title, notes)

        async def _tool_update_asana_task(
            meeting_id: str,
            task_gid: str,
            due_on: str | None = None,
            name: str | None = None,
            notes: str | None = None,
            assignee: str | None = None,
        ) -> str:
            return await update_asana_task(
                task_gid, due_on=due_on or None, name=name or None,
                notes=notes or None, assignee=assignee or None,
            )

        async def _tool_send_chat_message(meeting_id: str, message: str) -> str:
            if self._bot_status and self._bot_status.bot_id:
                await self._recall.send_chat_message(self._bot_status.bot_id, message)
                return "Message sent to meeting chat."
            return "Error: bot not active."

        async def _tool_log_decision(meeting_id: str, decision: str) -> str:
            context.add_decision(decision, meeting_id)
            return f"Decision logged: {decision}"

        async def _tool_search_past_meetings(meeting_id: str, query: str) -> str:
            return context.search_past_meetings(query)

        def _novnc_url() -> str:
            base = _SCREEN_API_URL.rsplit(":", 1)[0]
            return f"{base}:6080/vnc.html?autoconnect=1&resize=scale&view_only=0"

        async def _ensure_novnc_link(meeting_id: str) -> None:
            if meeting_id not in self._screen_link_sent:
                self._screen_link_sent.add(meeting_id)
                await self._send_chat(meeting_id, f"Screen sharing active: {_novnc_url()}")

        async def _tool_open_on_screen(meeting_id: str, url: str) -> str:
            from integrations.base import safe_post
            import urllib.parse as _urlparse
            # Use hostname as tab name so each domain gets its own tab
            tab = _urlparse.urlparse(url).netloc or url[:30]
            resp = await safe_post(f"{_SCREEN_API_URL}/open", {}, {"url": url, "tab": tab})
            if resp is None:
                return "Error: screen container unreachable."
            await _ensure_novnc_link(meeting_id)
            return f"Opened {url} on screen (tab: {tab})."

        async def _tool_act_on_screen(meeting_id: str, instruction: str) -> str:
            import json as _json
            import aiohttp
            self._screen_action_running = True
            summary = "Done."
            try:
                timeout = aiohttp.ClientTimeout(total=300, connect=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{_SCREEN_API_URL}/act",
                        json={"instruction": instruction},
                    ) as resp:
                        await _ensure_novnc_link(meeting_id)
                        async for line_bytes in resp.content:
                            line = line_bytes.decode().strip()
                            if not line.startswith("data:"):
                                continue
                            try:
                                event = _json.loads(line[5:].strip())
                            except _json.JSONDecodeError:
                                continue
                            etype = event.get("type")
                            if etype == "step" and _QUORUM_MODE == "active":
                                await self._speak_command(SpeakCommand(
                                    text=event.get("description", ""),
                                    meeting_id=meeting_id,
                                ))
                            elif etype == "done":
                                summary = event.get("summary", "Done.")
                                break
                            elif etype == "error":
                                summary = f"Screen action failed: {event.get('message', event.get('summary', 'unknown error'))}"
                                logger.error("act_on_screen error: %s", summary)
                                break
                            elif etype == "cancelled":
                                summary = "Task cancelled."
                                break
            finally:
                self._screen_action_running = False
            return summary

        async def _tool_render_visualization(
            meeting_id: str, description: str, data: str = ""
        ) -> str:
            import re as _re
            import aiohttp
            if not _OPENROUTER_KEY:
                return "Error: OPENROUTER_API_KEY not set."
            prompt = (
                f"Generate a beautiful self-contained single-file HTML visualization.\n"
                f"Description: {description}\n"
                f"Data: {data}\n\n"
                f"Requirements:\n"
                f"- Use Chart.js from https://cdn.jsdelivr.net/npm/chart.js\n"
                f"- Dark background (#0f0f0f), vibrant colours, clean professional look\n"
                f"- Title at top, responsive layout\n"
                f"- Return ONLY the complete HTML document, no explanation"
            )
            headers = {
                "Authorization": f"Bearer {_OPENROUTER_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": _OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(_OPENROUTER_URL, json=payload, headers=headers) as r:
                    resp_data = await r.json()
            html = resp_data["choices"][0]["message"]["content"]
            html = _re.sub(r"^```(?:html)?\s*", "", html.strip())
            html = _re.sub(r"\s*```$", "", html)
            from integrations.base import safe_post
            result = await safe_post(f"{_SCREEN_API_URL}/render_html", {}, {"html": html})
            if result is None:
                return "Error: screen container unreachable."
            await _ensure_novnc_link(meeting_id)
            return f"Visualization rendered on screen: {description}"

        agent.register_tools({
            "search_slack":          _tool_search_slack,
            "search_notion":         _tool_search_notion,
            "search_github":         _tool_search_github,
            "search_asana":          _tool_search_asana,
            "create_asana_task":     _tool_create_asana_task,
            "update_asana_task":     _tool_update_asana_task,
            "send_chat_message":     _tool_send_chat_message,
            "log_decision":          _tool_log_decision,
            "search_past_meetings":  _tool_search_past_meetings,
            "open_on_screen":        _tool_open_on_screen,
            "act_on_screen":         _tool_act_on_screen,
            "render_visualization":  _tool_render_visualization,
        })

        self._orchestrator.set_agent(agent)

        # Wire the segment callback into the manager
        self._manager.register_segment_callback(self.on_transcript_segment)

        logger.info("QBot initialised")

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

        logger.info("Q is starting up...")
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
        novnc_base = _SCREEN_API_URL.rsplit(":", 1)[0]
        novnc_link = f"{novnc_base}:6080/vnc.html?autoconnect=1&resize=scale&view_only=0"
        _join_msg = f"Hey! I'm Q 👋 — your AI meeting assistant.\n\nShared browser (for screen actions): {novnc_link}"
        self._bot_status = await self._recall.join_meeting(
            meeting_url, self._meeting_id, join_message=_join_msg
        )

        if self._bot_status.status == "error":
            logger.critical("Failed to join meeting — check RECALL_API_KEY and URL")
            await self.stop()
            return

        logger.info("Bot joining — bot_id=%s", self._bot_status.bot_id)

        # ── Step 4: Register session with manager + agent ────────────────
        self._manager.start_session(self._meeting_id, self._bot_status.bot_id)
        await self._orchestrator.start_meeting(self._meeting_id)

        # ── Step 4b: Greet the meeting and share the noVNC link ───────────
        # Poll until Recall confirms the bot is in the meeting (up to 30s)
        for _ in range(30):
            bot_status = await self._recall.get_bot_status(self._bot_status.bot_id)
            if bot_status.status == "active":
                break
            await asyncio.sleep(1)

        # Extra buffer — Google Meet chat API isn't available the instant the
        # bot status flips to active; give it a few more seconds to settle.
        await asyncio.sleep(5)
        await self._speaker.speak("Hey everyone, I'm Q, your AI meeting assistant. Ask me anything.")
        # The greeting chat message is sent by Recall.ai via on_bot_join
        # (configured at join time) — no manual send needed here.
        self._screen_link_sent.add(self._meeting_id)

        # ── Step 5: Keep running until interrupted ────────────────────────
        try:
            await asyncio.Event().wait()    # block forever
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """
        Leave the meeting and shut down all subsystems cleanly.
        """
        logger.info("Q is leaving the meeting...")

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
        logger.info("Q has left the meeting")

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

        # ── Cancel screen action if running and user says stop ────────────
        if self._screen_action_running:
            words = set(segment.text.lower().split())
            if words & _CANCEL_WORDS:
                await self._cancel_screen_action()
                return

        # ── AGENT HOOK — dispatch to orchestrator ─────────────────────────
        await self._orchestrator.process_segment(segment)
        # ─────────────────────────────────────────────────────────────────

    # ── Agent callbacks ───────────────────────────────────────────────────────

    async def _cancel_screen_action(self) -> None:
        from integrations.base import safe_post
        await safe_post(f"{_SCREEN_API_URL}/act/cancel", {}, {})
        logger.info("Screen action cancelled by user speech")
        if self._meeting_id:
            await self._speak_command(SpeakCommand(text="Stopping.", meeting_id=self._meeting_id))

    async def _speak_command(self, cmd: SpeakCommand) -> None:
        """Receive a SpeakCommand from the orchestrator and speak it aloud."""
        logger.info("Speaking: %r", cmd.text[:80])
        await self._speaker.speak(cmd.text)

    async def _integration_stub(self, req: ContextRequest) -> list[IntegrationResult]:
        """Live integration callback — queries GitHub, Notion, Slack, Asana."""
        return await integration_callback(req)

    async def _action_stub(self, req: ActionRequest) -> dict:
        """
        Stub action callback — replace with real actions-ui module.
        Logs the action until actions-ui branch is merged in.
        """
        logger.info("Action request (stub): type=%s params=%s", req.action_type, req.parameters)
        return {"status": "stub", "action_type": req.action_type}

    async def _send_chat(self, meeting_id: str, text: str) -> None:
        """Send a chat message into the meeting (URLs and full content preserved)."""
        if self._bot_status and self._bot_status.bot_id:
            await self._recall.send_chat_message(self._bot_status.bot_id, text)
        else:
            logger.warning("_send_chat: no active bot — dropping message")

    # ── Audio injection ───────────────────────────────────────────────────────

    async def _inject_audio(self, audio_bytes: bytes) -> None:
        """
        Inject MP3 audio bytes into the meeting via Recall.ai.

        This is the inject_callback passed to QSpeaker. Called
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
           Q
  AI Meeting Participant
=================================
"""


async def _main() -> None:
    """Async entry point — prompt for URL and start the bot."""
    bot = QBot()
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
