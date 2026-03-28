"""
test_integration.py — Final end-to-end integration test for the Quorum agent.

Instantiates QuorumOrchestrator with mock callbacks, feeds it the exact 5
transcript segments from the spec, and prints a pass/fail for each.

Run with:
    python3 test_integration.py
"""

import asyncio
import logging
import os
import time

# Configure logging before imports so module-level loggers pick it up
log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.WARNING),
    format="%(levelname)-8s %(name)s — %(message)s",
)

from agent import (
    ActionRequest,
    ContextRequest,
    IntegrationResult,
    QuorumOrchestrator,
    SpeakCommand,
    TranscriptSegment,
)


# ── Mock callbacks ────────────────────────────────────────────────────────────

spoken: list[SpeakCommand] = []
actions_dispatched: list[ActionRequest] = []
integrations_called: list[ContextRequest] = []


async def mock_speak(cmd: SpeakCommand) -> None:
    """Capture SpeakCommands instead of calling ElevenLabs."""
    spoken.append(cmd)


async def mock_integrate(req: ContextRequest) -> list[IntegrationResult]:
    """Return one dummy result for any integration query."""
    integrations_called.append(req)
    return [
        IntegrationResult(
            source="notion",
            title=f"Notion: {req.query}",
            url="https://notion.so/mock",
            summary=f"Context found for '{req.query}' in Notion.",
            raw_data={},
            timestamp=time.time(),
        )
    ]


async def mock_action(req: ActionRequest) -> dict:
    """Capture ActionRequests and return a mock success response."""
    actions_dispatched.append(req)
    return {"status": "ok", "action_type": req.action_type}


# ── Mock LLM (no network required) ───────────────────────────────────────────

async def mock_llm(prompt: str) -> str:
    """Return canned responses so the test runs fully offline."""
    p = prompt.lower()
    if "oauth" in p or "api" in p or "last week" in p:
        return "Last week the team decided to use OAuth 2.0 for all API endpoints."
    if "authentication" in p or "context found" in p:
        return "I found a Notion doc on authentication — OAuth 2.0 was chosen last sprint."
    return "I wasn't able to find relevant context for that."


# ── Test runner ───────────────────────────────────────────────────────────────

async def run() -> None:
    """Build the orchestrator, feed it 5 segments, report pass/fail."""

    orch = QuorumOrchestrator(
        speak_callback=mock_speak,
        integration_callback=mock_integrate,
        action_callback=mock_action,
    )

    # Patch LLM calls to be hermetic (no Hermes / OpenRouter needed)
    orch.call_llm = mock_llm  # type: ignore
    orch._context._call_llm = lambda p, timeout=10.0: mock_llm(p)  # type: ignore

    # Force ACTIVE mode so every intent fires without needing "Quorum" in text
    orch._mode.set_mode("active")

    meeting_id = "yhack-demo-001"
    await orch.start_meeting(meeting_id)

    # ── Exact segments from the spec ─────────────────────────────────────────

    segments = [
        TranscriptSegment(
            text="Let's talk about the new authentication feature",
            speaker="Speaker 0",
            timestamp=time.time(),
            is_final=True,
            meeting_id=meeting_id,
        ),
        TranscriptSegment(
            text="We'll go with OAuth 2.0, everyone agreed",
            speaker="Speaker 1",
            timestamp=time.time(),
            is_final=True,
            meeting_id=meeting_id,
        ),
        TranscriptSegment(
            text="Hey Quorum, what did we decide last week about the API?",
            speaker="Speaker 0",
            timestamp=time.time(),
            is_final=True,
            meeting_id=meeting_id,
        ),
        TranscriptSegment(
            text="Add that as a task for the backend team",
            speaker="Speaker 1",
            timestamp=time.time(),
            is_final=True,
            meeting_id=meeting_id,
        ),
        TranscriptSegment(
            text="Can you show me a chart of our sprint velocity?",
            speaker="Speaker 0",
            timestamp=time.time(),
            is_final=True,
            meeting_id=meeting_id,
        ),
    ]

    # (description, expected_outcome_key, expected_action_type_or_None)
    expectations = [
        ("topic surfaced",                     "integration_called",  None),
        ("decision logged",                    "decision_logged",     None),
        ("past meeting search + spoke",        "spoke",               None),
        ("action_callback: create_task",       "action_dispatched",   "create_task"),
        ("action_callback: generate_chart",    "action_dispatched",   "generate_chart"),
    ]

    print()
    print("=" * 70)
    print("  Quorum Agent — Final Integration Test")
    print("=" * 70)
    print()

    results: list[tuple[bool, str, str]] = []

    for seg, (description, outcome_key, expected_action) in zip(segments, expectations):
        # Snapshot counters before processing
        n_spoken    = len(spoken)
        n_actions   = len(actions_dispatched)
        n_integrations = len(integrations_called)
        n_decisions = len(orch._context.get_decisions(meeting_id))

        await orch.process_segment(seg)

        # Evaluate outcome
        if outcome_key == "integration_called":
            passed = len(integrations_called) > n_integrations
            detail = (
                f"integration query='{integrations_called[-1].query}'"
                if passed else "integration_callback NOT called"
            )
        elif outcome_key == "decision_logged":
            passed = len(orch._context.get_decisions(meeting_id)) > n_decisions
            decisions = orch._context.get_decisions(meeting_id)
            detail = (
                f"decision='{decisions[-1]}'"
                if passed else "decision NOT logged"
            )
        elif outcome_key == "spoke":
            passed = len(spoken) > n_spoken
            detail = (
                f"spoke='{spoken[-1].text[:60]}'"
                if passed else "speak_callback NOT called"
            )
        elif outcome_key == "action_dispatched":
            passed = len(actions_dispatched) > n_actions
            if passed and expected_action:
                actual = actions_dispatched[-1].action_type
                type_match = actual == expected_action
                detail = f"action_type='{actual}'" + ("" if type_match else f" (expected '{expected_action}')")
                passed = passed and type_match
            else:
                detail = "action_callback NOT called" if not passed else "dispatched"
        else:
            passed = False
            detail = "unknown expectation"

        results.append((passed, description, detail))

    # ── Print report ──────────────────────────────────────────────────────────

    total = len(results)
    total_pass = sum(1 for p, _, _ in results if p)

    for i, (passed, description, detail) in enumerate(results, 1):
        mark  = "PASS" if passed else "FAIL"
        arrow = "✓" if passed else "✗"
        seg_text = segments[i - 1].text[:48]
        print(f"  [{mark}] #{i} {seg_text!r}")
        print(f"         Expected : {description}")
        print(f"         Got      : {detail}")
        print()

    print("-" * 70)
    print(f"  Result: {total_pass}/{total} passed")
    print("-" * 70)
    print()

    if total_pass == total:
        print("  All tests passed. Agent is ready for integration.\n")
    else:
        print("  Some tests failed — check output above.\n")


if __name__ == "__main__":
    asyncio.run(run())
