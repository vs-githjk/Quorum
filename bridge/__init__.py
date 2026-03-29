"""
bridge/__init__.py — QuorumBridge: WebSocket server + callback wrappers.

Connects the Python orchestrator to the companion React app in real time.
Cards are pushed over WebSocket whenever Quorum surfaces context, takes an
action, or logs a decision.

Usage:

    from bridge import QuorumBridge, start_bridge
    from integrations import integration_callback
    from agent import QuorumOrchestrator

    async def main():
        bridge = QuorumBridge()
        runner = await start_bridge(bridge)          # starts ws://localhost:8765/ws

        orch = QuorumOrchestrator(
            speak_callback=bridge.wrap_speak(your_speak_fn),
            integration_callback=bridge.wrap_integration(integration_callback),
            action_callback=bridge.wrap_action(your_action_fn),
        )

        # ... run your meeting loop ...

        await runner.cleanup()                       # graceful shutdown
"""

import json
import logging

from aiohttp import web

from .mapper import action_to_card, decision_to_card, result_to_card

logger = logging.getLogger(__name__)


class QuorumBridge:
    """
    Maintains a set of connected WebSocket clients and broadcasts QuorumCard
    JSON whenever the orchestrator produces a result worth displaying.

    Wraps the three orchestrator callbacks (speak, integration, action) so
    the orchestrator itself needs zero changes.
    """

    def __init__(self) -> None:
        self._clients: set[web.WebSocketResponse] = set()
        # Tracks (meeting_id, url) pairs already broadcast — mirrors orchestrator dedup
        self._broadcast_urls: set[tuple[str, str]] = set()

    # ── WebSocket server ──────────────────────────────────────────────────────

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.info(
            "Companion connected — %d client(s) active", len(self._clients)
        )
        try:
            async for _ in ws:
                pass  # companion is receive-only; we ignore any incoming messages
        finally:
            self._clients.discard(ws)
            logger.info(
                "Companion disconnected — %d client(s) active", len(self._clients)
            )
        return ws

    def make_app(self) -> web.Application:
        """Return the aiohttp Application with the /ws route registered."""
        app = web.Application()
        app.router.add_get("/ws", self._handle_ws)
        return app

    # ── Broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(self, card: dict) -> None:
        """
        Send a QuorumCard dict as JSON to every connected companion client.

        Dead connections are silently removed.

        Args:
            card: QuorumCard-shaped dict from bridge.mapper.
        """
        if not self._clients:
            logger.debug("broadcast: no clients connected — card dropped")
            return

        payload = json.dumps(card)
        dead: set[web.WebSocketResponse] = set()

        for ws in self._clients:
            try:
                await ws.send_str(payload)
            except Exception as exc:
                logger.warning("Failed to send card to client: %s", exc)
                dead.add(ws)

        self._clients -= dead

    # ── Callback wrappers ─────────────────────────────────────────────────────

    def wrap_speak(self, fn):
        """
        Wrap speak_callback to detect decision confirmations and emit a card.

        Args:
            fn: The original async speak_callback (SpeakCommand) → None.

        Returns:
            Wrapped async function with the same signature.
        """
        bridge = self

        async def wrapped(cmd):
            await fn(cmd)
            if "logged that decision" in cmd.text.lower():
                card = decision_to_card(
                    meeting_id=cmd.meeting_id,
                    triggered_by=cmd.text,
                )
                await bridge.broadcast(card)

        return wrapped

    def wrap_integration(self, fn):
        """
        Wrap integration_callback to emit a card for each result returned.

        Args:
            fn: The original async integration_callback (ContextRequest) → list[IntegrationResult].

        Returns:
            Wrapped async function with the same signature.
        """
        bridge = self

        async def wrapped(req):
            results = await fn(req)
            for r in results:
                key = (req.meeting_id, r.url)
                if key in bridge._broadcast_urls:
                    continue
                bridge._broadcast_urls.add(key)
                card = result_to_card(
                    r,
                    triggered_by=req.query,
                    meeting_id=req.meeting_id,
                )
                await bridge.broadcast(card)
            return results

        return wrapped

    def wrap_action(self, fn):
        """
        Wrap action_callback to emit a card whenever an action is dispatched.

        Args:
            fn: The original async action_callback (ActionRequest) → dict.

        Returns:
            Wrapped async function with the same signature.
        """
        bridge = self

        async def wrapped(req):
            result = await fn(req)
            card = action_to_card(req, meeting_id=req.meeting_id)
            await bridge.broadcast(card)
            return result

        return wrapped


# ── Server startup helper ─────────────────────────────────────────────────────

async def start_bridge(
    bridge: QuorumBridge,
    host: str = "localhost",
    port: int = 8765,
) -> web.AppRunner:
    """
    Start the WebSocket server in the current asyncio event loop.

    Args:
        bridge: The QuorumBridge instance to serve.
        host:   Hostname to bind (default localhost).
        port:   Port to listen on (default 8765).

    Returns:
        AppRunner — call `await runner.cleanup()` to shut down gracefully.
    """
    runner = web.AppRunner(bridge.make_app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Quorum bridge listening on ws://%s:%d/ws", host, port)
    return runner
