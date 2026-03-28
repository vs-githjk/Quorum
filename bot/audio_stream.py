"""
bot/audio_stream.py — FastAPI webhook + WebSocket server for Quorum.

Recall.ai connects here to deliver real-time transcript data and bot
lifecycle events. Transcript data arrives over WebSocket; bot status
events arrive as HTTP POST webhooks.

# ── HOW TO TEST LOCALLY ────────────────────────────────────────────────────
#
#  1. Start ngrok in a separate terminal:
#         ngrok http 8000
#
#  2. Copy the https:// URL ngrok gives you (e.g. https://abc123.ngrok-free.app)
#     and set it in .env:
#         WEBHOOK_BASE_URL=https://abc123.ngrok-free.app
#
#  3. Start this server:
#         uvicorn bot.audio_stream:app --port 8000 --reload
#     or run via main_bot.py (which starts uvicorn automatically).
#
#  4. Confirm the server is reachable:
#         curl https://abc123.ngrok-free.app/health
#     Expected: {"status":"ok","bot_active":false,"meeting_id":null,"segments_received":0}
#
#  5. When a bot joins a meeting, Recall.ai will:
#     - Connect to wss://abc123.ngrok-free.app/ws/transcript (WebSocket)
#     - POST to https://abc123.ngrok-free.app/webhook/bot_status (HTTP)
#
# ── NOTE ON TRANSCRIPT DELIVERY ───────────────────────────────────────────
#
#  Recall.ai delivers transcripts over WebSocket (not HTTP POST).
#  The bot is configured with recording_config.realtime_endpoints type=websocket.
#  Recall.ai connects TO our /ws/transcript endpoint as a WebSocket client.
# ──────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Callable

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from bot import BotStatus, TranscriptSegment

logger = logging.getLogger(__name__)


class AudioStreamManager:
    """
    Central state manager for an active meeting session.

    Holds references to the active bot/meeting, the registered transcript
    callback, and running counters for observability.

    One instance is created at startup and shared across all FastAPI routes
    via dependency injection through the get_app() factory.
    """

    def __init__(self) -> None:
        """Initialise with no active session."""
        self.bot_id: str | None = None
        self.meeting_id: str | None = None
        self.segments_received: int = 0
        self._on_segment: Callable | None = None
        logger.debug("AudioStreamManager created")

    def register_segment_callback(self, callback: Callable) -> None:
        """
        Register the async callable that receives each TranscriptSegment.

        Called once at startup by main_bot.py before any meeting starts.

        Args:
            callback: Async callable with signature (TranscriptSegment) -> None.
        """
        self._on_segment = callback
        logger.info("Segment callback registered: %s", callback.__name__)

    def start_session(self, meeting_id: str, bot_id: str) -> None:
        """
        Initialise state for a new meeting session.

        Args:
            meeting_id: Unique meeting session ID.
            bot_id:     Recall.ai bot UUID for this session.
        """
        self.meeting_id = meeting_id
        self.bot_id = bot_id
        self.segments_received = 0
        logger.info(
            "Session started — meeting_id=%s bot_id=%s", meeting_id, bot_id
        )

    def end_session(self) -> None:
        """
        Clear active session state after the bot leaves or the call ends.
        """
        logger.info(
            "Session ended — meeting_id=%s segments=%d",
            self.meeting_id,
            self.segments_received,
        )
        self.bot_id = None
        self.meeting_id = None

    @property
    def bot_active(self) -> bool:
        """Return True if a meeting session is currently active."""
        return self.bot_id is not None


def get_app(manager: AudioStreamManager) -> FastAPI:
    """
    FastAPI application factory.

    Creates and returns a FastAPI app with the AudioStreamManager injected
    into all routes. Call this once in main_bot.py to wire everything together.

    Args:
        manager: The shared AudioStreamManager instance.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Quorum Bot Server")

    # ── WebSocket: /ws/transcript ─────────────────────────────────────────
    # Recall.ai connects here as a WebSocket CLIENT and pushes transcript
    # events. We accept the connection, receive JSON frames, parse each
    # transcript event, and dispatch to the registered on_segment callback.

    @app.websocket("/ws/transcript")
    async def ws_transcript(websocket: WebSocket) -> None:
        """
        Recall.ai real-time transcript WebSocket endpoint.

        Recall.ai opens this connection when the bot joins a meeting and
        streams transcript.data events as JSON until the meeting ends.

        Each message is parsed immediately and dispatched to the registered
        segment callback. Returns 200 (101 Switching Protocols) immediately
        on connect — never blocks.
        """
        await websocket.accept()
        client = websocket.client
        logger.info("WebSocket connected — client=%s", client)

        try:
            async for raw in websocket.iter_text():
                try:
                    data = json.loads(raw)
                    await _handle_transcript_event(data, manager)
                except json.JSONDecodeError:
                    logger.warning("ws_transcript: invalid JSON — %r", raw[:120])
                except Exception:
                    logger.error(
                        "ws_transcript: error handling message:\n%s",
                        traceback.format_exc(),
                    )
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected — client=%s", client)
        except Exception:
            logger.error(
                "ws_transcript: unexpected error:\n%s", traceback.format_exc()
            )

    # ── POST /webhook/bot_status ──────────────────────────────────────────

    @app.post("/webhook/bot_status")
    async def webhook_bot_status(request: Request) -> JSONResponse:
        """
        Receive Recall.ai bot lifecycle events via HTTP POST.

        Responds with 200 immediately before processing so Recall.ai never
        times out waiting for a response. Processing happens in a background
        task.

        Expected payload keys: bot_id, status_change (with code field).
        """
        try:
            body = await request.json()
        except Exception:
            logger.warning("webhook_bot_status: could not parse body")
            return JSONResponse({"ok": True})

        # Process async — never block the webhook response
        asyncio.create_task(_handle_status_event(body, manager))
        return JSONResponse({"ok": True})

    # ── GET /health ───────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> JSONResponse:
        """
        Health check endpoint.

        Returns current server state. Use this to confirm ngrok is tunnelling
        correctly before starting a meeting.
        """
        return JSONResponse({
            "status": "ok",
            "bot_active": manager.bot_active,
            "meeting_id": manager.meeting_id,
            "segments_received": manager.segments_received,
        })

    return app


# ── Event handlers ────────────────────────────────────────────────────────────

async def _handle_transcript_event(
    data: dict, manager: AudioStreamManager
) -> None:
    """
    Parse a transcript.data WebSocket event from Recall.ai and dispatch it.

    Recall.ai wraps Deepgram's response in an event envelope. This function
    handles both the envelope format and direct Deepgram payloads.

    Args:
        data:    Parsed JSON from the WebSocket message.
        manager: The shared AudioStreamManager instance.
    """
    # Recall.ai payload shape (deepgram_streaming provider):
    # {
    #   "event": "transcript.data",
    #   "data": {
    #     "data": {
    #       "words": [{"text": "Hey,", ...}, {"text": "Quorum.", ...}],
    #       "language_code": "en",
    #       "participant": {"id": "...", "name": "Speaker 0", ...},
    #       "is_final": true   (may be present)
    #     }
    #   }
    # }

    event_type = data.get("event", "")
    if event_type and event_type != "transcript.data":
        logger.debug("Ignoring non-transcript event: %s", event_type)
        return

    # Unwrap double-nested data.data
    outer = data.get("data", {})
    payload = outer.get("data", outer)

    # Build text by joining words array
    words = payload.get("words", [])
    if words:
        text = " ".join(w.get("text", "") for w in words).strip()
    else:
        # Fallback: direct text/transcript fields
        text = payload.get("transcript", "") or payload.get("text", "")
        if isinstance(text, dict):
            text = text.get("text", "")

    if not text or not text.strip():
        logger.debug("Empty transcript payload — skipping")
        return

    # Speaker from participant field
    participant = payload.get("participant", {})
    raw_speaker = (
        participant.get("name")
        or participant.get("id")
        or payload.get("speaker", 0)
    )
    # Normalise to "Speaker N" format
    try:
        speaker = f"Speaker {int(raw_speaker)}"
    except (TypeError, ValueError):
        speaker = str(raw_speaker) if raw_speaker else "Speaker 0"

    # Finality — word-based Recall.ai events are always final.
    # Fall back to True if the field is absent (deepgram_streaming provider
    # does not include is_final in the word-based payload).
    is_final = bool(
        payload.get("is_final", payload.get("final", True))
    )

    meeting_id = manager.meeting_id or "unknown"

    segment = TranscriptSegment(
        text=text.strip(),
        speaker=speaker,
        timestamp=time.time(),
        is_final=is_final,
        meeting_id=meeting_id,
    )

    manager.segments_received += 1
    logger.info("[%s] %s: %s", meeting_id, speaker, text)

    # Dispatch to registered callback
    if manager._on_segment is not None:
        if asyncio.iscoroutinefunction(manager._on_segment):
            asyncio.create_task(manager._on_segment(segment))
        else:
            manager._on_segment(segment)
    else:
        logger.warning("No segment callback registered — dropping segment")


def _extract_deepgram_text(payload: dict) -> str:
    """
    Extract transcript text from a raw Deepgram-format payload.

    Deepgram nests text inside channel.alternatives[0].transcript.
    This handles the case where Recall.ai passes through the raw Deepgram
    response without flattening it.

    Args:
        payload: Raw payload dict from the WebSocket message.

    Returns:
        Transcript text string, or empty string if not found.
    """
    try:
        return (
            payload["channel"]["alternatives"][0]["transcript"]
        )
    except (KeyError, IndexError, TypeError):
        return ""


async def _handle_status_event(body: dict, manager: AudioStreamManager) -> None:
    """
    Process a Recall.ai bot status change webhook event.

    Args:
        body:    Parsed JSON from the POST body.
        manager: The shared AudioStreamManager instance.
    """
    try:
        # Recall.ai status webhook format:
        # {"event": "bot.status_change", "data": {"bot": {"id": "..."}, "status": {"code": "..."}}}
        event_data = body.get("data", body)
        bot_info = event_data.get("bot", {})
        status_info = event_data.get("status", {})

        bot_id = bot_info.get("id", body.get("bot_id", "unknown"))
        code = status_info.get("code", body.get("status", "unknown"))

        logger.info("Bot status event — bot_id=%s code=%s", bot_id, code)

        if code in ("in_call_not_recording", "in_call_recording"):
            logger.info("Quorum is live in the meeting — bot_id=%s", bot_id)

        elif code in ("done", "call_ended", "fatal", "error"):
            logger.info(
                "Meeting ended — code=%s bot_id=%s", code, bot_id
            )
            manager.end_session()

    except Exception:
        logger.error(
            "_handle_status_event error:\n%s", traceback.format_exc()
        )


# ── Module-level app for uvicorn direct launch ────────────────────────────────
# Provides a default app instance so `uvicorn bot.audio_stream:app` works.
# main_bot.py uses get_app(manager) to create a properly wired instance.

_default_manager = AudioStreamManager()
app = get_app(_default_manager)


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    print("\n=== AudioStreamManager smoke test ===\n")

    # Verify AudioStreamManager state transitions
    mgr = AudioStreamManager()
    assert not mgr.bot_active
    print("[PASS] Initial state: bot_active=False")

    mgr.start_session("mtg-001", "bot-abc")
    assert mgr.bot_active
    assert mgr.meeting_id == "mtg-001"
    print("[PASS] start_session sets bot_active=True")

    mgr.end_session()
    assert not mgr.bot_active
    print("[PASS] end_session sets bot_active=False")

    # Verify callback registration
    async def dummy_callback(seg: TranscriptSegment) -> None:
        pass

    mgr.register_segment_callback(dummy_callback)
    assert mgr._on_segment is dummy_callback
    print("[PASS] register_segment_callback stores callback")

    # Verify FastAPI app builds and health route exists
    test_app = get_app(mgr)
    routes = [r.path for r in test_app.routes]
    assert "/health" in routes
    assert "/ws/transcript" in routes
    assert "/webhook/bot_status" in routes
    print("[PASS] FastAPI app has /health, /ws/transcript, /webhook/bot_status")

    print("\n=== All checks passed ===")
    print("\nTo run the live server:")
    print("  uvicorn bot.audio_stream:app --port 8000 --reload")
    print("Then visit http://localhost:8000/health\n")
