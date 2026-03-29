"""
run.py — Quorum entry point.

Starts the WebSocket bridge then runs an interactive loop that feeds
transcript segments to the orchestrator. Cards are pushed over
ws://localhost:8765/ws to the companion React app in real time.

Usage:

    python run.py

The companion app should be running at http://localhost:5173 (npm run dev).
Type transcript lines at the prompt. Empty line to quit.

Environment:
    Copy .env.example → .env and fill in your API keys before running.
"""

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
load_dotenv()  # must be before any integrations import

from agent import QuorumOrchestrator
from agent.orchestrator import SpeakCommand, ActionRequest, TranscriptSegment
from bridge import QuorumBridge, start_bridge
from integrations import integration_callback
from integrations.asana import create_asana_task

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MEETING_ID = "demo-meeting-001"


# ── Default speak / action callbacks ─────────────────────────────────────────

async def speak_callback(cmd: SpeakCommand) -> None:
    print(f"\n[Quorum] {cmd.text}\n")


async def action_callback(req: ActionRequest) -> dict:
    print(f"\n[Action] {req.action_type} — {req.parameters}")

    if req.action_type == "create_task":
        title = req.parameters.get("title") or req.context[:80]
        notes = f"Created by Quorum during meeting {req.meeting_id}. Context: {req.context}"
        task  = await create_asana_task(title=title, notes=notes)
        if task:
            print(f"[Asana] Task created: {task.get('permalink_url', '')}")
            return {"status": "ok", "task_gid": task.get("gid"), "url": task.get("permalink_url")}
        else:
            print("[Asana] Task creation failed — check token and workspace GID")
            return {"status": "error"}

    return {"status": "ok"}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    bridge = QuorumBridge()
    runner = await start_bridge(bridge)
    logger.info("Bridge ready — open the companion app, then start typing.")

    orch = QuorumOrchestrator(
        speak_callback=bridge.wrap_speak(speak_callback),
        integration_callback=bridge.wrap_integration(integration_callback),
        action_callback=bridge.wrap_action(action_callback),
    )

    print("\nQuorum is listening. Type transcript lines, empty line to quit.\n")

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                line = await loop.run_in_executor(None, input, "> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not line.strip():
                break
            await orch.process_segment(TranscriptSegment(
                text=line.strip(),
                speaker="Speaker 0",
                timestamp=time.time(),
                is_final=True,
                meeting_id=MEETING_ID,
            ))
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()
        logger.info("Bridge shut down.")


if __name__ == "__main__":
    asyncio.run(main())
