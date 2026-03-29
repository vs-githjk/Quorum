"""
agent/orchestrator.py — QOrchestrator (the heart of Quorum)

Wires together ModeManager, QAgent, and MeetingContext into a single
async processing loop. Receives TranscriptSegments from voice-bot, decides
whether and how to act, and dispatches to QAgent for agentic tool-calling.

All cross-branch contracts are defined here as dataclasses so teammates have
a single authoritative source of truth for the interfaces.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .context import MeetingContext
from .mode import ModeManager

logger = logging.getLogger(__name__)

# ── Timing constants ──────────────────────────────────────────────────────────

_ACTIVE_MODE_COOLDOWN = 15.0   # seconds between Q utterances in active mode

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
        text:       What Q should say aloud.
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
# Helpers
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


def _strip_q_trigger(text: str) -> str:
    """
    Remove the Q trigger phrase from a segment, returning the remainder.

    Normalises punctuation (commas, periods around the trigger) before matching
    so "Hey, Q." and "Hey Q" both strip cleanly.

    e.g. "Hey, Q. Could you check my Slack?" → "Could you check my Slack?"
         "Hey Q"                              → ""
    """
    from .mode import Q_TRIGGERS
    normalised = re.sub(r"\s+", " ", re.sub(r"[,\.!]+", " ", text)).strip()
    lower = normalised.lower()

    for trigger in sorted(Q_TRIGGERS, key=len, reverse=True):
        prefix_talking = f"i am talking to you {trigger}"
        if lower.startswith(prefix_talking):
            return normalised[len(prefix_talking):].strip()
        if lower.startswith(trigger):
            return normalised[len(trigger):].strip()

    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# QOrchestrator
# ═════════════════════════════════════════════════════════════════════════════

class QOrchestrator:
    """
    The central intelligence loop for Quorum.

    Receives TranscriptSegments, debounces them, checks mode and exchange
    state, and dispatches to QAgent for agentic tool-calling + LLM responses.

    All I/O with other branches is through the three async callbacks passed
    at construction time.
    """

    def __init__(
        self,
        speak_callback: Callable,
        integration_callback: Callable,
        action_callback: Callable,
        agent=None,
        chat_callback: Callable | None = None,
    ) -> None:
        """
        Initialise the orchestrator.

        Args:
            speak_callback:       async (SpeakCommand) → None
            integration_callback: async (ContextRequest) → list[IntegrationResult]
            action_callback:      async (ActionRequest) → dict
            agent:                QAgent instance (can also be set later via set_agent).
            chat_callback:        optional async (meeting_id, text) → None
                                  Called with the full response (URLs preserved) for
                                  meeting chat. Only called when content differs from
                                  the spoken (clean) version.
        """
        self._speak    = speak_callback
        self._integrate = integration_callback
        self._act      = action_callback
        self._agent    = agent
        self._chat     = chat_callback

        self._mode    = ModeManager()
        self._context = MeetingContext()

        self._active_meetings: set[str] = set()

        # Debounce — accumulate segments until speech settles before processing
        self._segment_buffers: dict[str, list] = {}
        self._debounce_tasks:  dict[str, asyncio.Task] = {}

        # Exchange state — IDLE or ENGAGED per meeting
        self._exchange_state:  dict[str, str]           = {}
        self._exchange_timers: dict[str, asyncio.Task]  = {}

        # Cooldown — prevents Q speaking too frequently in ACTIVE mode
        self._last_spoken: dict[str, float] = {}

        logger.info("QOrchestrator ready — mode=%s", self._mode.get_mode())

    def set_agent(self, agent) -> None:
        """Wire in a QAgent after construction (avoids circular-init issues)."""
        self._agent = agent

    # ── Public API ────────────────────────────────────────────────────────────

    async def start_meeting(self, meeting_id: str) -> None:
        """
        Initialise context for a new meeting session.

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
        buffered per-meeting and debounced. When the timer fires, all
        accumulated text is joined into one synthetic segment and handed
        to _process_flushed.

        Args:
            segment: A TranscriptSegment from voice-bot.
        """
        if not segment.is_final:
            return

        mid = segment.meeting_id

        # Reset the idle timer only if an exchange is already active —
        # do NOT create a new exchange from arbitrary speech.
        if self._get_exchange_state(mid) == _EXCHANGE_ENGAGED:
            self._set_exchange_engaged(mid)

        if mid not in self._segment_buffers:
            self._segment_buffers[mid] = []
        self._segment_buffers[mid].append(segment)

        existing = self._debounce_tasks.pop(mid, None)
        if existing and not existing.done():
            existing.cancel()

        timeout = _DEBOUNCE_COMPLETE if _looks_complete(segment.text) else _DEBOUNCE_FRAGMENT
        self._debounce_tasks[mid] = asyncio.create_task(
            self._debounce_fire(mid, timeout)
        )

    async def _debounce_fire(self, meeting_id: str, timeout: float) -> None:
        """Sleep then flush the segment buffer."""
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        segments = self._segment_buffers.pop(meeting_id, [])
        if not segments:
            return

        combined_text = " ".join(s.text.strip() for s in segments)
        flushed = TranscriptSegment(
            text=combined_text,
            speaker=segments[-1].speaker,
            timestamp=segments[-1].timestamp,
            is_final=True,
            meeting_id=meeting_id,
        )
        logger.debug(
            "[%s] Debounce flushed %d segment(s): %r",
            meeting_id, len(segments), combined_text[:80],
        )
        await self._process_flushed(flushed)

    async def _process_flushed(self, segment: TranscriptSegment) -> None:
        """
        Run the Q decision loop on a debounce-flushed segment.

        Flow:
            1. Store segment in MeetingContext.
            2. If addressed to Q: engage exchange, strip trigger.
               If not addressed and not engaged and mode != ACTIVE: drop.
            3. Cooldown check.
            4. Delegate to QAgent.run() → speak result.
        """
        mid = segment.meeting_id
        self._context.add_segment(segment)

        addressed = self._mode.is_addressed_to_q(segment.text)
        if addressed:
            self._set_exchange_engaged(mid)
            text = _strip_q_trigger(segment.text)
            if not text or len(text.split()) <= 1:
                await self._speak_and_record(SpeakCommand(
                    text="Hey! What do you need?",
                    meeting_id=mid,
                ))
                return
        else:
            from .mode import ACTIVE
            if (
                self._get_exchange_state(mid) != _EXCHANGE_ENGAGED
                and self._mode.get_mode() != ACTIVE
            ):
                return
            text = segment.text

        if self._on_cooldown(mid):
            logger.debug("[%s] On cooldown — skipping agent call", mid)
            return

        if self._agent is None:
            logger.error("[%s] No QAgent configured — dropping segment", mid)
            return

        response = await self._agent.run(text, mid, segment.speaker)
        if response:
            await self._speak_and_record(SpeakCommand(text=response.spoken, meeting_id=mid))
            if response.chat and self._chat:
                await self._chat(mid, response.chat)

    # ── Cooldown + speak helpers ──────────────────────────────────────────────

    def _on_cooldown(self, meeting_id: str) -> bool:
        """Return True if Q has spoken within _ACTIVE_MODE_COOLDOWN seconds."""
        last = self._last_spoken.get(meeting_id, 0.0)
        return (time.time() - last) < _ACTIVE_MODE_COOLDOWN

    def _record_speak(self, meeting_id: str) -> None:
        """Record that Q just spoke, resetting the cooldown timer."""
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

        Called whenever Q speaks or is directly addressed. After
        _EXCHANGE_IDLE_TIMEOUT seconds without activity the meeting
        reverts to IDLE.
        """
        self._exchange_state[meeting_id] = _EXCHANGE_ENGAGED
        existing = self._exchange_timers.pop(meeting_id, None)
        if existing and not existing.done():
            existing.cancel()
        self._exchange_timers[meeting_id] = asyncio.create_task(
            self._exchange_idle_after(meeting_id)
        )
        logger.debug("[%s] Exchange state → ENGAGED", meeting_id)

    async def _exchange_idle_after(self, meeting_id: str) -> None:
        """Transition the meeting to IDLE after the inactivity timeout."""
        await asyncio.sleep(_EXCHANGE_IDLE_TIMEOUT)
        self._exchange_state[meeting_id] = _EXCHANGE_IDLE
        self._exchange_timers.pop(meeting_id, None)
        logger.info(
            "[%s] Exchange state → IDLE (no activity for %.0fs)",
            meeting_id, _EXCHANGE_IDLE_TIMEOUT,
        )

    # ── Response cleanup ─────────────────────────────────────────────────────

    @staticmethod
    def _clean_response(text: str) -> str:
        """
        Strip any trailing SKIP marker the LLM may have appended.

        e.g. "I can't open emails. SKIP" → "I can't open emails."
        """
        cleaned = re.sub(r"[\s\n]+SKIP\s*$", "", text.strip(), flags=re.IGNORECASE)
        return cleaned.strip()
