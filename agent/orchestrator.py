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
import re
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
_OPENROUTER_MODEL = "nousresearch/hermes-3-llama-3.1-70b"
_LLM_TIMEOUT = 0.3          # seconds before Hermes is considered unavailable (fail fast)
_ACTIVE_MODE_COOLDOWN  = 15.0  # seconds between Quorum utterances in active mode
_CLARIFICATION_TIMEOUT = 30.0  # seconds before a pending clarification is abandoned
_OFFER_TIMEOUT         = 20.0  # seconds before a pending proactive offer is abandoned
_PROACTIVE = os.getenv("QUORUM_PROACTIVE", "false").lower() == "true"

# Debounce — wait for speech to settle before processing
_DEBOUNCE_COMPLETE = 0.8   # seconds — segment ends with .!? (complete sentence)
_DEBOUNCE_FRAGMENT = 2.2   # seconds — segment looks mid-sentence

# Exchange state — tracks whether an active conversation is happening
_EXCHANGE_IDLE_TIMEOUT = 20.0   # seconds of silence before going idle
_EXCHANGE_IDLE    = "idle"
_EXCHANGE_ENGAGED = "engaged"

# Words that commonly appear at the end of incomplete phrases
_CONTINUATION_ENDINGS: frozenset[str] = frozenset({
    "the", "a", "an", "in", "on", "to", "of", "with", "for", "from",
    "at", "by", "into", "about", "and", "or", "but", "so",
    "my", "your", "our", "their", "its", "this", "that", "which",
    "just", "also", "even", "still", "i", "we", "you", "they", "he", "she", "it",
})

# Question starters — if segment begins with one of these but has no ?,
# the speaker is almost certainly still mid-sentence
_QUESTION_STARTERS: frozenset[str] = frozenset({
    "what", "where", "when", "why", "how", "which", "who", "whose", "whom",
    "could", "can", "would", "will", "is", "are", "do", "does", "did",
})

# Affirmative / negative keywords for resolving pending proactive offers
_AFFIRMATIVE = {"yes", "yeah", "yep", "sure", "go ahead", "do it", "please", "yup"}
_NEGATIVE    = {"no", "nope", "don't", "skip", "cancel", "stop", "nevermind"}

_LLM_SYSTEM_PROMPT = (
    "You are Quorum, an AI meeting participant. Be helpful and concise. "
    "Keep all spoken responses under 2 sentences. "
    "You CAN: answer questions from meeting context, search GitHub PRs/issues, "
    "Notion docs, Slack messages, and Asana tasks, log decisions, and create tasks. "
    "You CANNOT: open apps, browse the internet, access personal calendars, "
    "send emails, or do anything outside the meeting context. "
    "If asked to do something outside your capabilities, say so clearly in one sentence. "
    "Never make up capabilities you do not have. Reply only in English."
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

def _looks_complete(text: str) -> bool:
    """
    Heuristic: does this transcript segment look like a finished utterance?

    Returns False (fragment) when:
    - 2 words or fewer
    - Starts with a question word but doesn't end with ?
    - Ends with a preposition, article, or conjunction

    Returns True (complete) when:
    - Ends with . ! ?
    - None of the fragment signals above apply
    """
    stripped = text.strip()
    if not stripped:
        return True
    words = stripped.split()
    if len(words) <= 2:
        return False
    if stripped[-1] in ".!?":
        return True
    if words[0].lower() in _QUESTION_STARTERS:
        return False
    if words[-1].lower() in _CONTINUATION_ENDINGS:
        return False
    return True


def _strip_quorum_trigger(text: str) -> str:
    """
    Remove the Quorum trigger phrase from a segment, returning the remainder.

    Normalises punctuation (commas, periods around the trigger) before matching
    so "Hey, Q." and "Hey Q" both strip cleanly.

    e.g. "Hey, Coron. Could you check my Slack?" → "Could you check my Slack?"
         "Hey, Quorum."                           → ""
         "Coram can you open Notion?"             → "can you open Notion?"
         "I am talking to you, Coram. Help?"      → "Help?"
    """
    from .mode import QUORUM_TRIGGERS
    # Normalise: replace commas/periods/exclamations with spaces (keep ?)
    # then collapse runs of whitespace so "Hey, Q." → "Hey Q"
    normalised = re.sub(r"\s+", " ", re.sub(r"[,\.!]+", " ", text)).strip()
    lower = normalised.lower()

    for trigger in sorted(QUORUM_TRIGGERS, key=len, reverse=True):
        prefix_talking = f"i am talking to you {trigger}"
        if lower.startswith(prefix_talking):
            return normalised[len(prefix_talking):].strip()
        if lower.startswith(trigger):
            return normalised[len(trigger):].strip()

    return text.strip()


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

        # Debounce — accumulate segments until speech settles before processing
        self._segment_buffers: dict[str, list] = {}
        self._debounce_tasks: dict[str, asyncio.Task] = {}

        # Exchange state — IDLE or ENGAGED per meeting
        self._exchange_state: dict[str, str] = {}
        self._exchange_timers: dict[str, asyncio.Task] = {}

        # Cooldown — prevents Quorum speaking too frequently in ACTIVE mode
        self._last_spoken: dict[str, float] = {}

        # Proactive offer state — "should I create a task?" waiting for yes/no
        # Structure: {action, original_req, offered_at}
        self._pending_offers: dict[str, dict] = {}

        # Clarification state — "which PR?" waiting for an answer
        # Structure: {intent_type, original_segment, asked_at, question}
        self._pending_clarifications: dict[str, dict] = {}

        # Pending clarification state — keyed by meeting_id
        # Structure: {intent_type, original_segment, asked_at, question}
        self._pending_clarifications: dict[str, dict] = {}

        # Proactive offer state — keyed by meeting_id
        # Structure: {action, original_req, offered_at}
        self._pending_offers: dict[str, dict] = {}

        # Cooldown tracking — keyed by meeting_id, value is last-spoken timestamp
        self._last_spoken: dict[str, float] = {}

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
        Entry point called by voice-bot on every Deepgram event.

        Non-final segments are dropped immediately. Final segments are
        buffered per-meeting and debounced: a short pause (_DEBOUNCE_COMPLETE)
        after a sentence-ending segment, a longer pause (_DEBOUNCE_FRAGMENT)
        after a mid-sentence fragment.  When the timer fires, all accumulated
        text is joined into one synthetic segment and handed to _process_flushed.

        Args:
            segment: A TranscriptSegment from voice-bot.
        """
        if not segment.is_final:
            return

        mid = segment.meeting_id

        # Accumulate this segment
        if mid not in self._segment_buffers:
            self._segment_buffers[mid] = []
        self._segment_buffers[mid].append(segment)

        # Cancel any pending debounce for this meeting
        existing = self._debounce_tasks.pop(mid, None)
        if existing and not existing.done():
            existing.cancel()

        # Choose timeout: complete sentence → short wait, fragment → longer wait
        timeout = _DEBOUNCE_COMPLETE if _looks_complete(segment.text) else _DEBOUNCE_FRAGMENT
        self._debounce_tasks[mid] = asyncio.create_task(
            self._debounce_fire(mid, timeout)
        )

    async def _debounce_fire(self, meeting_id: str, timeout: float) -> None:
        """
        Sleep for `timeout` seconds, then flush the segment buffer and process.

        Cancelled immediately when a new segment arrives for the same meeting —
        the newer segment reschedules with its own timeout.
        """
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        segments = self._segment_buffers.pop(meeting_id, [])
        if not segments:
            return

        # Join all buffered text into one synthetic segment
        combined_text = " ".join(s.text.strip() for s in segments)
        flushed = TranscriptSegment(
            text=combined_text,
            speaker=segments[-1].speaker,
            timestamp=segments[-1].timestamp,
            is_final=True,
            meeting_id=meeting_id,
        )
        logger.debug("[%s] Debounce flushed %d segment(s): %r", meeting_id, len(segments), combined_text[:80])
        await self._process_flushed(flushed)

    async def _process_flushed(self, segment: TranscriptSegment) -> None:
        """
        Run the full Quorum decision loop on a debounce-flushed segment.

        Decision flow:
            1.  Store segment in MeetingContext.
            2.  Resolve any pending offer or clarification.
            3.  Detect intent.
            4.  If NONE → respond only when engaged or directly addressed.
            5.  Check mode gate.
            6.  Dispatch based on intent type.

        Args:
            segment: A combined, final TranscriptSegment ready for processing.
        """
        mid = segment.meeting_id
        logger.debug("[%s] Processing: %r", mid, segment.text[:80])

        # ── Step 1: store in context ──────────────────────────────────────────
        self._context.add_segment(segment)

        # ── Step 2: resolve pending offer or clarification ───────────────────
        offer = self._pending_offers.get(mid)
        if offer:
            if time.time() - offer["offered_at"] > _OFFER_TIMEOUT:
                logger.info("[%s] Pending offer timed out — clearing", mid)
                del self._pending_offers[mid]
            else:
                lowered = segment.text.lower()
                if any(kw in lowered for kw in _AFFIRMATIVE):
                    logger.info("[%s] Offer accepted — dispatching action", mid)
                    req = offer["original_req"]
                    self._context.add_action(req)
                    result = await self._act(req)
                    logger.info("[%s] Deferred action result: %s", mid, result)
                    del self._pending_offers[mid]
                    return
                elif any(kw in lowered for kw in _NEGATIVE):
                    logger.info("[%s] Offer declined — clearing", mid)
                    del self._pending_offers[mid]
                    return
                # Ambiguous — fall through to normal intent detection

        pending = self._pending_clarifications.get(mid)
        if pending:
            if time.time() - pending["asked_at"] > _CLARIFICATION_TIMEOUT:
                logger.info("[%s] Clarification timed out — clearing", mid)
                del self._pending_clarifications[mid]
            else:
                await self._resolve_clarification(segment, pending)
                del self._pending_clarifications[mid]
                return

        # ── Step 3: strip trigger + detect intent ────────────────────────────
        from .intent import NONE, ACTION_TASK, ACTION_PR, ACTION_CHART, DECISION, QUESTION, TOPIC_MENTION

        addressed = self._mode.is_addressed_to_quorum(segment.text)
        if addressed:
            self._set_exchange_engaged(mid)
            # Strip "Hey, Coron." etc. so intent is detected on the real request
            remaining = _strip_quorum_trigger(segment.text)
            if remaining and len(remaining.split()) > 1:
                segment = TranscriptSegment(
                    text=remaining,
                    speaker=segment.speaker,
                    timestamp=segment.timestamp,
                    is_final=True,
                    meeting_id=segment.meeting_id,
                )
            else:
                # Pure address with no content — freeform greeting
                await self._handle_freeform(segment)
                return

        intent = self._intent.detect(segment)
        logger.info(
            "[%s] Intent=%s confidence=%.2f topic=%r requires_llm=%s",
            mid,
            intent.type,
            intent.confidence,
            intent.extracted_topic,
            intent.requires_llm,
        )

        # ── Step 4: handle NONE intent ────────────────────────────────────────
        if intent.type == NONE:
            if addressed or self._get_exchange_state(mid) == _EXCHANGE_ENGAGED:
                # In an active exchange — pass to LLM to decide if worth responding
                await self._handle_freeform(segment)
            else:
                logger.debug("[%s] Intent=NONE, not engaged — staying quiet", mid)
            return

        # ── Step 5: check mode gate ───────────────────────────────────────────
        from .mode import ACTIVE
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
        mid   = segment.meeting_id
        topic = intent.extracted_topic or segment.text
        req   = ActionRequest(
            action_type="create_task",
            parameters={"title": topic, "source_text": segment.text},
            context=segment.text,
            meeting_id=mid,
        )

        if _PROACTIVE and not self._on_cooldown(mid):
            self._pending_offers[mid] = {
                "action": "create_task", "original_req": req, "offered_at": time.time(),
            }
            logger.info("[%s] Proactive: offering to create task %r", mid, topic)
            await self._speak_and_record(SpeakCommand(
                text="I could create a task for that — want me to?",
                meeting_id=mid, priority="normal",
            ))
            return

        self._context.add_action(req)
        logger.info("[%s] Dispatching create_task: %r", mid, topic)
        result = await self._act(req)
        logger.info("[%s] create_task result: %s", mid, result)
        await self._speak_and_record(SpeakCommand(
            text="Got it — I've added that as a task.",
            meeting_id=mid, priority="normal",
        ))

    async def _handle_action_pr(self, segment: TranscriptSegment, intent) -> None:
        """
        Dispatch a pull_pr ActionRequest when someone asks to pull up a PR.

        If no PR reference (number or name) is found in the transcript, asks
        a clarifying question and stores a pending state. The next segment
        resolves the clarification and dispatches the action.

        Args:
            segment: The triggering transcript segment.
            intent:  The detected Intent with extracted_topic.
        """
        mid = segment.meeting_id

        # Ask for clarification if no PR reference was found in the transcript
        if not intent.extracted_topic:
            self._pending_clarifications[mid] = {
                "intent_type": "ACTION_PR",
                "original_segment": segment,
                "asked_at": time.time(),
                "question": "Which PR are you referring to?",
            }
            logger.info("[%s] ACTION_PR — no reference, asking for clarification", mid)
            await self._speak(SpeakCommand(
                text="Which PR are you referring to?",
                meeting_id=mid, priority="normal",
            ))
            return

        topic = intent.extracted_topic
        req = ActionRequest(
            action_type="pull_pr",
            parameters={"reference": topic, "source_text": segment.text},
            context=segment.text,
            meeting_id=mid,
        )
        self._context.add_action(req)
        logger.info("[%s] Dispatching pull_pr: %r", mid, topic)
        result = await self._act(req)
        logger.info("[%s] pull_pr result: %s", mid, result)
        await self._speak_and_record(SpeakCommand(
            text=f"Pulling up the PR for {topic} now.",
            meeting_id=mid, priority="normal",
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

        await self._speak_and_record(SpeakCommand(
            text="Generating that chart now.",
            meeting_id=segment.meeting_id, priority="normal",
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

        await self._speak_and_record(SpeakCommand(
            text="Noted — I've logged that decision.",
            meeting_id=segment.meeting_id, priority="normal",
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
            # Pull live context from all integrations — let the router narrow it down
            results = await self._integrate(ContextRequest(
                query=topic,
                sources=["github", "notion", "slack", "asana"],
                meeting_id=mid,
            ))
            # Deduplicate — don't surface the same URL twice in a meeting
            results = [r for r in results if not self._context.is_already_surfaced(r.url, mid)]
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
        await self._speak_and_record(SpeakCommand(
            text=response, meeting_id=mid, priority="normal",
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

        # Don't interrupt ambient conversation too frequently —
        # but never block when the user is actively talking to the bot.
        if self._on_cooldown(mid) and self._get_exchange_state(mid) != _EXCHANGE_ENGAGED:
            logger.info("[%s] Skipping topic surface — on cooldown (not engaged)", mid)
            return

        results = await self._integrate(ContextRequest(
            query=topic,
            sources=["github", "notion", "slack"],
            meeting_id=mid,
        ))

        if not results:
            from .mode import ACTIVE as _ACTIVE
            addressed = self._mode.is_addressed_to_quorum(segment.text)
            if self._mode.get_mode() == _ACTIVE or addressed:
                if addressed:
                    self._set_exchange_engaged(mid)
                # Single LLM call — no chained freeform to avoid double-call latency
                recent = self._context.get_recent_transcript(mid, n=5)
                prompt = (
                    f"Someone in a meeting said: \"{segment.text}\"\n\n"
                    f"Recent conversation:\n{recent}\n\n"
                    f"Respond helpfully in 1–2 sentences, or reply SKIP if no response is needed."
                )
                response = self._clean_response(await self.call_llm(prompt))
                if response and response.upper() != "SKIP":
                    await self._speak_and_record(SpeakCommand(text=response, meeting_id=mid, priority="normal"))
            else:
                logger.debug("[%s] No integration results for topic %r — staying quiet", mid, topic)
            return

        # Deduplicate — don't surface the same URL twice in one meeting
        results = [r for r in results if not self._context.is_already_surfaced(r.url, mid)]
        if not results:
            logger.debug("[%s] All results already surfaced for topic %r — skipping", mid, topic)
            return

        for r in results:
            self._context.add_surfaced_result(r, mid)

        results_text = "\n".join(f"- [{r.source}] {r.title}: {r.summary}" for r in results)
        recent = self._context.get_recent_transcript(mid, n=3)
        prompt = (
            f"The meeting just mentioned '{topic}'.\n\n"
            f"Recent transcript:\n{recent}\n\n"
            f"Relevant context found:\n{results_text}\n\n"
            f"If this context would genuinely help the meeting right now, "
            f"reply with a 1–2 sentence spoken summary starting with 'I found'. "
            f"If it would not help, reply with exactly: SKIP"
        )

        response = self._clean_response(await self.call_llm(prompt))

        if not response or response.upper() == "SKIP":
            logger.info("[%s] LLM decided topic context not worth surfacing", mid)
            return

        logger.info("[%s] Surfacing context for topic %r", mid, topic)
        await self._speak_and_record(SpeakCommand(
            text=response, meeting_id=mid, priority="normal",
        ))

    async def _handle_freeform(self, segment: TranscriptSegment) -> None:
        """
        Handle a segment with no recognised intent when Quorum is engaged.

        Sends the text to the LLM with the recent transcript as context.
        The LLM replies with SKIP if the message doesn't need a response,
        or with a 1–2 sentence spoken answer if it does.

        Skipped entirely when Quorum is on cooldown (spoke recently) to
        prevent back-to-back responses from split segments.

        Args:
            segment: The transcript segment to respond to.
        """
        mid = segment.meeting_id

        if self._on_cooldown(mid):
            logger.debug("[%s] Freeform: on cooldown — skipping", mid)
            return

        recent = self._context.get_recent_transcript(mid, n=6)
        prompt = (
            f"You are Quorum, an AI participant in a meeting. Someone just said: \"{segment.text}\"\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"If this is directed at you or is a question you can genuinely help with, "
            f"respond in 1–2 sentences.\n"
            f"If it is general meeting conversation that does not need your input, "
            f"reply with exactly: SKIP"
        )
        response = self._clean_response(await self.call_llm(prompt))
        if not response or response.upper() == "SKIP":
            logger.debug("[%s] LLM decided freeform not worth responding to", mid)
            return
        logger.info("[%s] Freeform LLM response: %r", mid, response[:80])
        await self._speak_and_record(SpeakCommand(text=response, meeting_id=mid, priority="normal"))

    # ── Cooldown + speak helpers ──────────────────────────────────────────────

    def _on_cooldown(self, meeting_id: str) -> bool:
        """Return True if Quorum has spoken within _ACTIVE_MODE_COOLDOWN seconds."""
        last = self._last_spoken.get(meeting_id, 0.0)
        return (time.time() - last) < _ACTIVE_MODE_COOLDOWN

    def _record_speak(self, meeting_id: str) -> None:
        """Record that Quorum just spoke, resetting the cooldown timer."""
        self._last_spoken[meeting_id] = time.time()

    async def _speak_and_record(self, cmd: SpeakCommand) -> None:
        """Speak, update the cooldown timer, and mark the exchange as engaged."""
        await self._speak(cmd)
        self._record_speak(cmd.meeting_id)
        self._set_exchange_engaged(cmd.meeting_id)

    # ── Exchange state ────────────────────────────────────────────────────────

    def _get_exchange_state(self, meeting_id: str) -> str:
        """Return the current exchange state for a meeting (IDLE or ENGAGED)."""
        return self._exchange_state.get(meeting_id, _EXCHANGE_IDLE)

    def _set_exchange_engaged(self, meeting_id: str) -> None:
        """
        Mark the meeting as ENGAGED and reset the idle timer.

        Called whenever Quorum speaks or is directly addressed. After
        _EXCHANGE_IDLE_TIMEOUT seconds without speaking, the meeting
        reverts to IDLE and Quorum stops responding to ambient chatter.
        """
        self._exchange_state[meeting_id] = _EXCHANGE_ENGAGED
        # Cancel any running idle countdown
        existing = self._exchange_timers.pop(meeting_id, None)
        if existing and not existing.done():
            existing.cancel()
        # Schedule idle transition
        self._exchange_timers[meeting_id] = asyncio.create_task(
            self._exchange_idle_after(meeting_id)
        )
        logger.debug("[%s] Exchange state → ENGAGED", meeting_id)

    async def _exchange_idle_after(self, meeting_id: str) -> None:
        """Transition the meeting to IDLE after the inactivity timeout."""
        await asyncio.sleep(_EXCHANGE_IDLE_TIMEOUT)
        self._exchange_state[meeting_id] = _EXCHANGE_IDLE
        self._exchange_timers.pop(meeting_id, None)
        logger.info("[%s] Exchange state → IDLE (no activity for %.0fs)", meeting_id, _EXCHANGE_IDLE_TIMEOUT)

    # ── Clarification resolver ────────────────────────────────────────────────

    async def _resolve_clarification(
        self,
        answer_segment: TranscriptSegment,
        pending: dict,
    ) -> None:
        """
        Re-dispatch the original intent using the user's clarifying answer.

        Builds a synthetic Intent with the answer text as the extracted topic
        and routes it to the appropriate handler.
        """
        from .intent import Intent, ACTION_PR, ACTION_TASK

        mid         = answer_segment.meeting_id
        intent_type = pending["intent_type"]
        answer_text = answer_segment.text.strip()

        logger.info("[%s] Resolving clarification for %s: %r", mid, intent_type, answer_text)

        resolved = Intent(
            type=intent_type,
            confidence=0.90,
            extracted_topic=answer_text,
            raw_text=answer_text,
            requires_llm=False,
        )

        if intent_type == ACTION_PR:
            await self._handle_action_pr(answer_segment, resolved)
        elif intent_type == ACTION_TASK:
            await self._handle_action_task(answer_segment, resolved)
        else:
            logger.warning("[%s] No clarification resolver for %s", mid, intent_type)

    # ── Response cleanup ─────────────────────────────────────────────────────

    @staticmethod
    def _clean_response(text: str) -> str:
        """
        Strip any trailing SKIP marker the LLM may have appended to a real sentence.

        The LLM is instructed to reply with exactly 'SKIP' when it has nothing
        to say, but sometimes appends 'SKIP' to an otherwise valid response.
        This strips the trailing marker so the real content gets spoken.

        e.g. "I can't open emails. SKIP" → "I can't open emails."
        """
        cleaned = re.sub(r"[\s\n]+SKIP\s*$", "", text.strip(), flags=re.IGNORECASE)
        return cleaned.strip()

    # ── Cooldown + proactive speak helpers ───────────────────────────────────

    def _on_cooldown(self, meeting_id: str) -> bool:
        """Return True if Quorum has spoken within _ACTIVE_MODE_COOLDOWN seconds."""
        last = self._last_spoken.get(meeting_id, 0.0)
        return (time.time() - last) < _ACTIVE_MODE_COOLDOWN

    def _record_speak(self, meeting_id: str) -> None:
        """Record that Quorum just spoke (for cooldown tracking)."""
        self._last_spoken[meeting_id] = time.time()

    async def _speak_and_record(self, cmd: SpeakCommand) -> None:
        """Speak and update the cooldown timer for this meeting."""
        await self._speak(cmd)
        self._record_speak(cmd.meeting_id)

    # ── Clarification helpers ─────────────────────────────────────────────────

    async def _resolve_clarification(
        self,
        answer_segment: TranscriptSegment,
        pending: dict,
    ) -> None:
        """
        Re-dispatch the original intent using the clarifying answer.

        Takes the pending intent type and re-runs the appropriate handler,
        substituting the answer segment's text as the extracted topic.

        Args:
            answer_segment: The segment containing the user's answer.
            pending:        The stored clarification dict.
        """
        from .intent import Intent, ACTION_PR, ACTION_TASK

        mid          = answer_segment.meeting_id
        intent_type  = pending["intent_type"]
        answer_text  = answer_segment.text.strip()

        logger.info(
            "[%s] Resolving clarification for %s with answer: %r",
            mid, intent_type, answer_text,
        )

        # Build a synthetic Intent using the answer as the topic
        resolved_intent = Intent(
            type=intent_type,
            confidence=0.90,
            extracted_topic=answer_text,
            raw_text=answer_text,
            requires_llm=False,
        )

        if intent_type == ACTION_PR:
            await self._handle_action_pr(answer_segment, resolved_intent)
        elif intent_type == ACTION_TASK:
            await self._handle_action_task(answer_segment, resolved_intent)
        else:
            logger.warning(
                "[%s] No clarification resolver for intent type %s", mid, intent_type
            )

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
