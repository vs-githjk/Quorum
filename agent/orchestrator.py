"""
agent/orchestrator.py — QuorumOrchestrator (the heart of Quorum)

Wires together ModeManager, IntentDetector, and MeetingContext into a single
async processing loop. Receives TranscriptSegments from voice-bot, decides
whether and how to act, and dispatches to integrations / actions / TTS.

All cross-branch contracts are defined here as dataclasses so teammates have
a single authoritative source of truth for the interfaces.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import aiohttp

from .context import MeetingContext
from .intent import IntentDetector
from .mode import ModeManager

logger = logging.getLogger(__name__)

# ── LLM configuration ─────────────────────────────────────────────────────────

_HERMES_HOST = os.getenv("HERMES_HOST", "http://localhost:11434")
_HERMES_URL = f"{_HERMES_HOST}/api/generate"
_HERMES_MODEL = os.getenv("HERMES_MODEL", "hermes3")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = "nousresearch/hermes-3-llama-3.1-405b:free"
_LLM_TIMEOUT = 2.0          # seconds before Hermes is considered unavailable
_LLM_SYSTEM_PROMPT = (
    "You are Quorum, an AI meeting participant. You are helpful, concise, "
    "and only speak when you have something genuinely useful to add. "
    "Keep all spoken responses under 2 sentences. You are currently in a meeting."
)


# ═════════════════════════════════════════════════════════════════════════════
# Cross-branch interface dataclasses
# All other branches must conform to these exact shapes.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TranscriptSegment:
    """
    Sent by voice-bot to the orchestrator for every Deepgram transcript event.

    Fields:
        text:       The spoken words from this segment.
        speaker:    Diarization label from Deepgram e.g. "Speaker 0".
        timestamp:  Unix timestamp of when the segment was captured.
        is_final:   False = interim (speculative) result; True = committed.
                    The orchestrator ignores non-final segments.
        meeting_id: Unique identifier for this meeting session.
    """
    text: str
    speaker: str
    timestamp: float
    is_final: bool
    meeting_id: str


@dataclass
class SpeakCommand:
    """
    Sent by the orchestrator to voice-bot to trigger ElevenLabs TTS.

    Fields:
        text:       What Quorum should say aloud.
        meeting_id: The active meeting session.
        priority:   'high' = interrupt current speaker; 'normal' = wait for silence.
    """
    text: str
    meeting_id: str
    priority: str = "normal"    # "high" | "normal"


@dataclass
class ContextRequest:
    """
    Sent by the orchestrator to the integrations layer to fetch relevant context.

    Fields:
        query:       The topic or keyword to search for.
        sources:     Which integrations to query e.g. ["github", "notion", "slack"].
        meeting_id:  The active meeting session.
        max_results: Maximum number of results to return per source.
    """
    query: str
    sources: list[str]
    meeting_id: str
    max_results: int = 3


@dataclass
class IntegrationResult:
    """
    Returned by the integrations layer in response to a ContextRequest.

    Fields:
        source:    "github" | "notion" | "slack" | "asana"
        title:     Human-readable title of the result.
        url:       Direct link to the resource.
        summary:   1–2 sentence description of the content.
        raw_data:  Full payload for downstream use (actions, rendering).
        timestamp: Unix timestamp of the result (creation or retrieval time).
    """
    source: str
    title: str
    url: str
    summary: str
    raw_data: dict
    timestamp: float


@dataclass
class ActionRequest:
    """
    Sent by the orchestrator to the actions layer to trigger an action.

    Fields:
        action_type: "create_task" | "pull_pr" | "generate_chart"
        parameters:  Action-specific key/value pairs (e.g. task title, PR number).
        context:     The transcript text that triggered this action.
        meeting_id:  The active meeting session.
    """
    action_type: str
    parameters: dict
    context: str
    meeting_id: str


# ═════════════════════════════════════════════════════════════════════════════
# QuorumOrchestrator
# ═════════════════════════════════════════════════════════════════════════════

class QuorumOrchestrator:
    """
    The central intelligence loop for Quorum.

    Receives TranscriptSegments, classifies intent, checks whether to respond,
    dispatches to integrations / actions as needed, and triggers TTS via a
    speak callback.

    All I/O with other branches is through the three async callbacks passed
    at construction time — the orchestrator itself has no direct network
    dependencies beyond the LLM calls.
    """

    def __init__(
        self,
        speak_callback: Callable,
        integration_callback: Callable,
        action_callback: Callable,
    ) -> None:
        """
        Initialise the orchestrator with its three external callbacks.

        Args:
            speak_callback:       async (SpeakCommand) → None
                                  Called to make Quorum speak via ElevenLabs.
            integration_callback: async (ContextRequest) → list[IntegrationResult]
                                  Called to fetch context from GitHub / Notion / Slack.
            action_callback:      async (ActionRequest) → dict
                                  Called to execute actions (create task, pull PR, etc.).
        """
        self._speak = speak_callback
        self._integrate = integration_callback
        self._act = action_callback

        self._mode = ModeManager()
        self._intent = IntentDetector()
        self._context = MeetingContext()

        self._active_meetings: set[str] = set()

        logger.info(
            "QuorumOrchestrator ready — mode=%s", self._mode.get_mode()
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def start_meeting(self, meeting_id: str) -> None:
        """
        Initialise context for a new meeting session.

        Safe to call multiple times for the same meeting_id (idempotent).

        Args:
            meeting_id: Unique identifier for the meeting.
        """
        if meeting_id in self._active_meetings:
            logger.warning("start_meeting called again for %s — ignoring", meeting_id)
            return
        self._active_meetings.add(meeting_id)
        logger.info("Meeting started: %s", meeting_id)

    async def end_meeting(self, meeting_id: str) -> None:
        """
        Finalise a meeting: persist transcript, generate summary, log stats.

        Args:
            meeting_id: The meeting to end.
        """
        self._active_meetings.discard(meeting_id)
        await self._context.end_meeting(meeting_id)
        logger.info("Meeting ended: %s", meeting_id)

    async def process_segment(self, segment: TranscriptSegment) -> None:
        """
        Run the full Quorum decision loop for one incoming transcript segment.

        This is the main entry point called by voice-bot on every Deepgram event.
        Non-final (interim) segments are dropped immediately — the orchestrator
        only acts on committed transcript text.

        Decision flow:
            1.  Drop interim segments.
            2.  Store segment in MeetingContext.
            3.  Detect intent.
            4.  If NONE → stay quiet.
            5.  Ask ModeManager whether to respond.
            6.  If mode says no → log and return.
            7.  Dispatch based on intent type.

        Args:
            segment: A TranscriptSegment from voice-bot.
        """
        # ── Step 1: skip non-final ────────────────────────────────────────────
        if not segment.is_final:
            logger.debug("Skipping interim segment for %s", segment.meeting_id)
            return

        mid = segment.meeting_id
        logger.debug(
            "[%s] Processing: %r", mid, segment.text[:80]
        )

        # ── Step 2: store in context ──────────────────────────────────────────
        self._context.add_segment(segment)

        # ── Step 3: detect intent ─────────────────────────────────────────────
        intent = self._intent.detect(segment)
        logger.info(
            "[%s] Intent=%s confidence=%.2f topic=%r requires_llm=%s",
            mid,
            intent.type,
            intent.confidence,
            intent.extracted_topic,
            intent.requires_llm,
        )

        # ── Step 4: bail on NONE — but first check if Quorum was addressed ──────
        from .intent import NONE, ACTION_TASK, ACTION_PR, ACTION_CHART, DECISION, QUESTION, TOPIC_MENTION

        if intent.type == NONE:
            # Even with no recognized intent, respond if directly addressed
            if self._mode.is_addressed_to_quorum(segment.text):
                logger.info("[%s] Addressed with no specific intent — acknowledging", mid)
                await self._speak(SpeakCommand(
                    text="I'm here. What do you need?",
                    meeting_id=mid,
                    priority="normal",
                ))
            else:
                logger.debug("[%s] Intent=NONE — staying quiet", mid)
            return

        # ── Step 5: check mode ────────────────────────────────────────────────
        has_context = intent.type != NONE
        if not self._mode.should_respond(segment.text, has_relevant_context=has_context):
            logger.info(
                "[%s] Mode=%s — suppressing response to intent=%s",
                mid,
                self._mode.get_mode(),
                intent.type,
            )
            return

        # ── Step 6: dispatch by intent ────────────────────────────────────────

        if intent.type == ACTION_TASK:
            await self._handle_action_task(segment, intent)

        elif intent.type == ACTION_PR:
            await self._handle_action_pr(segment, intent)

        elif intent.type == ACTION_CHART:
            await self._handle_action_chart(segment, intent)

        elif intent.type == DECISION:
            await self._handle_decision(segment, intent)

        elif intent.type == QUESTION:
            await self._handle_question(segment, intent)

        elif intent.type == TOPIC_MENTION:
            await self._handle_topic_mention(segment, intent)

    # ── Intent handlers ───────────────────────────────────────────────────────

    async def _handle_action_task(self, segment: TranscriptSegment, intent) -> None:
        """
        Dispatch a create_task ActionRequest when someone asks to create a task.

        Args:
            segment: The triggering transcript segment.
            intent:  The detected Intent with extracted_topic.
        """
        topic = intent.extracted_topic or segment.text
        req = ActionRequest(
            action_type="create_task",
            parameters={"title": topic, "source_text": segment.text},
            context=segment.text,
            meeting_id=segment.meeting_id,
        )
        self._context.add_action(req)
        logger.info("[%s] Dispatching create_task: %r", segment.meeting_id, topic)
        result = await self._act(req)
        logger.info("[%s] create_task result: %s", segment.meeting_id, result)

        await self._speak(SpeakCommand(
            text=f"Got it — I've added that as a task.",
            meeting_id=segment.meeting_id,
            priority="normal",
        ))

    async def _handle_action_pr(self, segment: TranscriptSegment, intent) -> None:
        """
        Dispatch a pull_pr ActionRequest when someone asks to pull up a PR.

        Args:
            segment: The triggering transcript segment.
            intent:  The detected Intent with extracted_topic.
        """
        topic = intent.extracted_topic or "unknown"
        req = ActionRequest(
            action_type="pull_pr",
            parameters={"reference": topic, "source_text": segment.text},
            context=segment.text,
            meeting_id=segment.meeting_id,
        )
        self._context.add_action(req)
        logger.info("[%s] Dispatching pull_pr: %r", segment.meeting_id, topic)
        result = await self._act(req)
        logger.info("[%s] pull_pr result: %s", segment.meeting_id, result)

        await self._speak(SpeakCommand(
            text=f"Pulling up the PR for {topic} now.",
            meeting_id=segment.meeting_id,
            priority="normal",
        ))

    async def _handle_action_chart(self, segment: TranscriptSegment, intent) -> None:
        """
        Dispatch a generate_chart ActionRequest when a visualisation is requested.

        Args:
            segment: The triggering transcript segment.
            intent:  The detected Intent with extracted_topic.
        """
        topic = intent.extracted_topic or segment.text
        req = ActionRequest(
            action_type="generate_chart",
            parameters={"topic": topic, "source_text": segment.text},
            context=segment.text,
            meeting_id=segment.meeting_id,
        )
        self._context.add_action(req)
        logger.info("[%s] Dispatching generate_chart: %r", segment.meeting_id, topic)
        result = await self._act(req)
        logger.info("[%s] generate_chart result: %s", segment.meeting_id, result)

        await self._speak(SpeakCommand(
            text=f"Generating that chart now.",
            meeting_id=segment.meeting_id,
            priority="normal",
        ))

    async def _handle_decision(self, segment: TranscriptSegment, intent) -> None:
        """
        Log a detected decision and optionally confirm it aloud.

        Args:
            segment: The triggering transcript segment.
            intent:  The detected Intent with extracted_topic.
        """
        decision_text = intent.extracted_topic or segment.text
        self._context.add_decision(decision_text, segment.meeting_id)
        logger.info("[%s] Decision logged: %r", segment.meeting_id, decision_text)

        await self._speak(SpeakCommand(
            text=f"Noted — I've logged that decision.",
            meeting_id=segment.meeting_id,
            priority="normal",
        ))

    async def _handle_question(self, segment: TranscriptSegment, intent) -> None:
        """
        Answer a question by searching past meetings and/or calling the LLM.

        If requires_llm is False and a topic is available, searches past meeting
        history and speaks a summary. If requires_llm is True, also calls the
        LLM to reformulate the answer into natural language.

        Args:
            segment: The triggering transcript segment.
            intent:  The detected Intent with requires_llm and extracted_topic.
        """
        mid = segment.meeting_id
        topic = intent.extracted_topic or segment.text
        logger.info("[%s] Handling QUESTION about: %r", mid, topic)

        past = self._context.search_past_meetings(topic)

        if intent.requires_llm or past.startswith("No past"):
            # Pull live context from integrations as well
            results = await self._integrate(ContextRequest(
                query=topic,
                sources=["notion", "slack"],
                meeting_id=mid,
            ))
            for r in results:
                self._context.add_surfaced_result(r, mid)

            integration_text = (
                "\n".join(f"- {r.title}: {r.summary}" for r in results)
                if results else "No integration results found."
            )

            recent = self._context.get_recent_transcript(mid, n=5)
            prompt = (
                f"Someone in a meeting asked: \"{segment.text}\"\n\n"
                f"Recent transcript:\n{recent}\n\n"
                f"Past meeting context:\n{past}\n\n"
                f"Integration results:\n{integration_text}\n\n"
                f"Answer in 1–2 sentences as Quorum."
            )
            response = await self.call_llm(prompt)
        else:
            # Fast path: keyword search gave us enough to answer directly
            prompt = (
                f"Someone asked: \"{segment.text}\"\n\n"
                f"Relevant past meeting notes:\n{past}\n\n"
                f"Summarise the answer in 1 sentence as Quorum."
            )
            response = await self.call_llm(prompt)

        logger.info("[%s] Speaking question answer: %r", mid, response[:80])
        await self._speak(SpeakCommand(
            text=response,
            meeting_id=mid,
            priority="normal",
        ))

    async def _handle_topic_mention(self, segment: TranscriptSegment, intent) -> None:
        """
        Surface relevant context when a notable topic is mentioned.

        Calls the integrations layer for live context, formats a brief spoken
        summary (1–2 sentences), and speaks it if the LLM decides it is
        worth surfacing.

        Args:
            segment: The triggering transcript segment.
            intent:  The detected Intent with extracted_topic.
        """
        mid = segment.meeting_id
        topic = intent.extracted_topic or segment.text
        logger.info("[%s] Topic mentioned: %r — fetching context", mid, topic)

        results = await self._integrate(ContextRequest(
            query=topic,
            sources=["github", "notion", "slack"],
            meeting_id=mid,
        ))

        if not results:
            # If Quorum itself was addressed (e.g. "Hey Quorum"), acknowledge even
            # though there are no integration results to surface.
            if self._mode.is_addressed_to_quorum(segment.text):
                logger.info("[%s] Addressed with no integration context — acknowledging", mid)
                await self._speak(SpeakCommand(
                    text="I'm here. What do you need?",
                    meeting_id=mid,
                    priority="normal",
                ))
            else:
                logger.debug("[%s] No integration results for topic %r — staying quiet", mid, topic)
            return

        for r in results:
            self._context.add_surfaced_result(r, mid)

        # Ask the LLM whether this is actually worth saying aloud
        results_text = "\n".join(
            f"- [{r.source}] {r.title}: {r.summary}" for r in results
        )
        recent = self._context.get_recent_transcript(mid, n=3)
        prompt = (
            f"The meeting just mentioned '{topic}'.\n\n"
            f"Recent transcript:\n{recent}\n\n"
            f"Relevant context found:\n{results_text}\n\n"
            f"If this context would genuinely help the meeting right now, "
            f"reply with a 1–2 sentence spoken summary starting with 'I found'. "
            f"If it would not help, reply with exactly: SKIP"
        )

        response = await self.call_llm(prompt)

        if response.strip().upper() == "SKIP" or not response:
            logger.info("[%s] LLM decided topic context not worth surfacing", mid)
            return

        logger.info("[%s] Surfacing context for topic %r", mid, topic)
        await self._speak(SpeakCommand(
            text=response,
            meeting_id=mid,
            priority="normal",
        ))

    # ── LLM ──────────────────────────────────────────────────────────────────

    async def call_llm(self, prompt: str) -> str:
        """
        Call the LLM: try Hermes locally first, fall back to OpenRouter.

        Hermes has a 2-second timeout. If it is unavailable or too slow,
        OpenRouter (mistral-7b-instruct) is used instead.

        All calls share the same system prompt positioning Quorum as a concise
        meeting participant.

        Args:
            prompt: The user-facing prompt to complete.

        Returns:
            The LLM's response as a plain string. Returns a safe fallback
            message if both providers fail.
        """
        try:
            result = await self._call_hermes(prompt)
            logger.debug("LLM: Hermes responded")
            return result
        except Exception as exc:
            logger.warning("Hermes unavailable (%s) — falling back to OpenRouter", exc)

        try:
            result = await self._call_openrouter(prompt)
            logger.debug("LLM: OpenRouter responded")
            return result
        except Exception as exc:
            logger.error("OpenRouter also failed: %s", exc)
            return "I wasn't able to retrieve that information right now."

    async def _call_hermes(self, prompt: str) -> str:
        """
        POST to the local Hermes / Ollama endpoint.

        Args:
            prompt: Prompt text.

        Returns:
            Generated text.

        Raises:
            aiohttp.ClientError or asyncio.TimeoutError on failure.
        """
        payload = {
            "model": _HERMES_MODEL,
            "prompt": f"{_LLM_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=_LLM_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_HERMES_URL, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("response", "").strip()

    async def _call_openrouter(self, prompt: str) -> str:
        """
        POST to OpenRouter as LLM fallback.

        Args:
            prompt: Prompt text.

        Returns:
            Generated text.

        Raises:
            aiohttp.ClientError on failure, ValueError if API key is missing.
        """
        if not _OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY env var not set")

        headers = {
            "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": _OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL, json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()


# ═════════════════════════════════════════════════════════════════════════════
# Standalone integration test
# ═════════════════════════════════════════════════════════════════════════════

async def _run_test() -> None:
    """
    Full async integration test with mock callbacks.

    Feeds 5 TranscriptSegments through the orchestrator — one per intent type —
    and prints what action was taken for each.

    Run with:
        python3 -m agent.orchestrator
    """
    import types

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    # ── Mock callbacks ────────────────────────────────────────────────────────

    spoken: list[SpeakCommand] = []
    actions_dispatched: list[ActionRequest] = []
    integrations_called: list[ContextRequest] = []

    async def mock_speak(cmd: SpeakCommand) -> None:
        """Capture speak commands instead of calling ElevenLabs."""
        spoken.append(cmd)

    async def mock_integrate(req: ContextRequest) -> list[IntegrationResult]:
        """Return a single dummy result for any integration query."""
        integrations_called.append(req)
        return [
            IntegrationResult(
                source="notion",
                title=f"Notion doc: {req.query}",
                url="https://notion.so/mock",
                summary=f"Previously discussed {req.query} — see linked doc.",
                raw_data={},
                timestamp=time.time(),
            )
        ]

    async def mock_action(req: ActionRequest) -> dict:
        """Capture action requests and return a mock success payload."""
        actions_dispatched.append(req)
        return {"status": "ok", "action_type": req.action_type}

    # ── Patch LLM so test runs without Hermes / OpenRouter ───────────────────

    async def mock_llm(self_unused, prompt: str) -> str:
        """Return a canned LLM response based on prompt keywords."""
        p = prompt.lower()
        if "oauth" in p or "api" in p or "decided" in p:
            return "Last week the team decided to use OAuth 2.0 for all API endpoints."
        if "authentication" in p or "topic" in p or "found" in p:
            return "I found a Notion doc on authentication — the team chose OAuth 2.0 last sprint."
        return "I wasn't able to find relevant context for that."

    # ── Build orchestrator with mock callbacks ────────────────────────────────

    orch = QuorumOrchestrator(
        speak_callback=mock_speak,
        integration_callback=mock_integrate,
        action_callback=mock_action,
    )
    # Patch call_llm so the test is hermetic (no network needed)
    orch.call_llm = lambda prompt: mock_llm(None, prompt)  # type: ignore
    orch._context._call_llm = lambda p, timeout=10.0: mock_llm(None, p)  # type: ignore

    # Force ACTIVE mode so all segments trigger responses
    orch._mode.set_mode("active")

    meeting_id = "test-meeting-001"
    await orch.start_meeting(meeting_id)

    # ── Test segments ─────────────────────────────────────────────────────────

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

    expected = [
        ("TOPIC_MENTION",  "integration_called",  None,            None),
        ("DECISION",       "decision_logged",      None,            None),
        ("QUESTION",       "spoke",                None,            None),
        ("ACTION_TASK",    "action_dispatched",    "create_task",   None),
        ("ACTION_CHART",   "action_dispatched",    "generate_chart",None),
    ]

    print("\n=== QuorumOrchestrator integration test ===\n")

    results: list[tuple[str, bool, str]] = []

    # Snapshot state before each segment so we can detect what changed
    for seg, (exp_intent, exp_outcome, exp_action_type, _) in zip(segments, expected):
        spoken_before    = len(spoken)
        actions_before   = len(actions_dispatched)
        integrations_before = len(integrations_called)
        decisions_before = len(orch._context.get_decisions(meeting_id))

        await orch.process_segment(seg)

        spoke_now    = len(spoken) > spoken_before
        acted_now    = len(actions_dispatched) > actions_before
        integrated   = len(integrations_called) > integrations_before
        decided      = len(orch._context.get_decisions(meeting_id)) > decisions_before

        if exp_outcome == "action_dispatched":
            passed = acted_now
            last_action = actions_dispatched[-1].action_type if acted_now else "—"
            type_ok = (last_action == exp_action_type) if exp_action_type else True
            passed = passed and type_ok
            outcome_desc = f"action_callback called with action_type='{last_action}'"
        elif exp_outcome == "decision_logged":
            passed = decided
            outcome_desc = f"decision logged (total={len(orch._context.get_decisions(meeting_id))})"
        elif exp_outcome == "integration_called":
            passed = integrated
            outcome_desc = f"integration_callback called (query={integrations_called[-1].query!r})" if integrated else "integration NOT called"
        elif exp_outcome == "spoke":
            passed = spoke_now
            outcome_desc = f"speak_callback called: {spoken[-1].text!r}" if spoke_now else "speak NOT called"
        else:
            passed = False
            outcome_desc = "unknown expected outcome"

        results.append((exp_intent, passed, outcome_desc, seg.text))

    print(f"{'#':<3} {'Segment':<52} {'Intent':<15} {'Result':<10} Details")
    print("─" * 130)
    total_pass = 0
    for i, (exp_intent, passed, desc, text) in enumerate(results, 1):
        mark = "PASS" if passed else "FAIL"
        if passed:
            total_pass += 1
        print(f"{i:<3} {text[:50]:<52} {exp_intent:<15} [{mark}]     {desc}")

    print(f"\n{total_pass}/{len(results)} passed\n")


def test() -> None:
    """Entry point for direct module execution."""
    asyncio.run(_run_test())


if __name__ == "__main__":
    test()
