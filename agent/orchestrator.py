"""
agent/orchestrator.py — QuorumOrchestrator (the heart of Quorum)

Wires together ModeManager, MeetingContext, and a LangGraph agent into a
single async processing loop. Receives TranscriptSegments from voice-bot,
debounces them, checks mode/cooldown gates, and invokes the LangGraph
for intelligent responses.

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

from .context import MeetingContext
from .mode import ModeManager

logger = logging.getLogger(__name__)

# ── Timing constants ─────────────────────────────────────────────────────────

_ACTIVE_MODE_COOLDOWN  = 15.0  # seconds between Quorum utterances in active mode
_DEBOUNCE_COMPLETE     = 0.8   # seconds — segment ends with .!? (complete sentence)
_DEBOUNCE_FRAGMENT     = 2.2   # seconds — segment looks mid-sentence
_EXCHANGE_IDLE_TIMEOUT = 20.0  # seconds of silence before going idle
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
    """
    text: str
    meeting_id: str
    priority: str = "normal"    # "high" | "normal"


@dataclass
class ContextRequest:
    """
    Sent by the orchestrator to the integrations layer to fetch relevant context.
    """
    query: str
    sources: list[str]
    meeting_id: str
    max_results: int = 3


@dataclass
class IntegrationResult:
    """
    Returned by the integrations layer in response to a ContextRequest.
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
    """
    action_type: str
    parameters: dict
    context: str
    meeting_id: str


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _looks_complete(text: str) -> bool:
    """Heuristic: does this transcript segment look like a finished utterance?"""
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
    """
    from .mode import QUORUM_TRIGGERS
    normalised = re.sub(r"\s+", " ", re.sub(r"[,\.!]+", " ", text)).strip()
    lower = normalised.lower()

    for trigger in sorted(QUORUM_TRIGGERS, key=len, reverse=True):
        prefix_talking = f"i am talking to you {trigger}"
        if lower.startswith(prefix_talking):
            return normalised[len(prefix_talking):].strip()
        if lower.startswith(trigger):
            return normalised[len(trigger):].strip()

    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# QuorumOrchestrator
# ═════════════════════════════════════════════════════════════════════════════

class QuorumOrchestrator:
    """
    The central intelligence loop for Quorum.

    Receives TranscriptSegments, debounces them, checks mode/cooldown gates,
    and invokes the LangGraph agent for all response generation.
    """

    def __init__(
        self,
        speak_callback: Callable,
        integration_callback: Callable,
        action_callback: Callable,
    ) -> None:
        self._speak = speak_callback
        self._integrate = integration_callback
        self._act = action_callback

        self._mode = ModeManager()
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

        # Per-meeting conversation memory for LangGraph
        self._message_history: dict[str, list] = {}

        # Build the LangGraph
        from .graph import build_graph
        self._graph = build_graph(self._context, action_callback)

        logger.info(
            "QuorumOrchestrator ready — mode=%s (LangGraph)", self._mode.get_mode()
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def start_meeting(self, meeting_id: str) -> None:
        """Initialise context for a new meeting session."""
        if meeting_id in self._active_meetings:
            logger.warning("start_meeting called again for %s — ignoring", meeting_id)
            return
        self._active_meetings.add(meeting_id)
        logger.info("Meeting started: %s", meeting_id)

    async def end_meeting(self, meeting_id: str) -> None:
        """Finalise a meeting: persist transcript, generate summary, log stats."""
        self._active_meetings.discard(meeting_id)
        self._message_history.pop(meeting_id, None)
        await self._context.end_meeting(meeting_id)
        logger.info("Meeting ended: %s", meeting_id)

    async def process_segment(self, segment: TranscriptSegment) -> None:
        """
        Entry point called by voice-bot on every Deepgram event.

        Non-final segments are dropped. Final segments are buffered and
        debounced before processing.
        """
        if not segment.is_final:
            return

        mid = segment.meeting_id

        if mid not in self._segment_buffers:
            self._segment_buffers[mid] = []
        self._segment_buffers[mid].append(segment)

        # Cancel any pending debounce for this meeting
        existing = self._debounce_tasks.pop(mid, None)
        if existing and not existing.done():
            existing.cancel()

        timeout = _DEBOUNCE_COMPLETE if _looks_complete(segment.text) else _DEBOUNCE_FRAGMENT
        self._debounce_tasks[mid] = asyncio.create_task(
            self._debounce_fire(mid, timeout)
        )

    async def _debounce_fire(self, meeting_id: str, timeout: float) -> None:
        """Sleep for timeout, then flush the segment buffer and process."""
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
        logger.debug("[%s] Debounce flushed %d segment(s): %r", meeting_id, len(segments), combined_text[:80])
        await self._process_flushed(flushed)

    # ── Core processing (invokes LangGraph) ──────────────────────────────────

    async def _process_flushed(self, segment: TranscriptSegment) -> None:
        """
        Run the Quorum decision loop on a debounce-flushed segment.

        1. Store in context
        2. Check if addressed → strip trigger, set engaged
        3. Mode gate (ON_DEMAND requires being addressed)
        4. Cooldown gate
        5. Invoke LangGraph
        6. Speak if graph decided to
        """
        mid = segment.meeting_id

        # Store in context
        self._context.add_segment(segment)

        # Check if Quorum was addressed
        addressed = self._mode.is_addressed_to_quorum(segment.text)
        if addressed:
            self._set_exchange_engaged(mid)
            remaining = _strip_quorum_trigger(segment.text)
            if remaining and len(remaining.split()) > 1:
                segment = TranscriptSegment(
                    text=remaining,
                    speaker=segment.speaker,
                    timestamp=segment.timestamp,
                    is_final=True,
                    meeting_id=mid,
                )

        # Mode gate: in ON_DEMAND, only process if addressed or engaged
        exchange = self._get_exchange_state(mid)
        mode = self._mode.get_mode()
        if mode == "on_demand" and not addressed and exchange != _EXCHANGE_ENGAGED:
            logger.debug("[%s] ON_DEMAND, not addressed — staying quiet", mid)
            return

        # Cooldown: don't respond too frequently to ambient chatter
        if not addressed and exchange != _EXCHANGE_ENGAGED and self._on_cooldown(mid):
            logger.debug("[%s] On cooldown — skipping", mid)
            return

        # Invoke the LangGraph
        logger.info("[%s] Invoking LangGraph for: %r", mid, segment.text[:80])
        try:
            result = await self._graph.ainvoke({
                "segment_text": segment.text,
                "speaker": segment.speaker,
                "meeting_id": mid,
                "mode": mode,
                "is_addressed": addressed,
                "exchange_state": exchange,
                "recent_transcript": self._context.get_recent_transcript(mid, n=6),
                "messages": self._message_history.get(mid, []),
                "response_text": "",
                "should_speak": False,
                "decisions_logged": [],
            })
        except Exception as exc:
            logger.error("[%s] LangGraph invocation failed: %s", mid, exc, exc_info=True)
            if addressed:
                await self._speak(SpeakCommand(
                    text="Sorry, I ran into an issue. Could you repeat that?",
                    meeting_id=mid, priority="normal",
                ))
            return

        # Accumulate message history (keep last 20 messages)
        self._message_history[mid] = list(result.get("messages", []))[-20:]

        # Process decisions
        for dec in result.get("decisions_logged", []):
            self._context.add_decision(dec, mid)

        # Speak if the graph decided to
        if result.get("should_speak") and result.get("response_text"):
            text = _clean_response(result["response_text"])
            if text and text.upper() != "SKIP":
                logger.info("[%s] Speaking: %r", mid, text[:80])
                await self._speak_and_record(SpeakCommand(
                    text=text, meeting_id=mid, priority="normal",
                ))

    # ── Cooldown + speak helpers ─────────────────────────────────────────────

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

    # ── Exchange state ───────────────────────────────────────────────────────

    def _get_exchange_state(self, meeting_id: str) -> str:
        """Return the current exchange state for a meeting (IDLE or ENGAGED)."""
        return self._exchange_state.get(meeting_id, _EXCHANGE_IDLE)

    def _set_exchange_engaged(self, meeting_id: str) -> None:
        """Mark the meeting as ENGAGED and reset the idle timer."""
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
        logger.info("[%s] Exchange state → IDLE (no activity for %.0fs)", meeting_id, _EXCHANGE_IDLE_TIMEOUT)


# ═════════════════════════════════════════════════════════════════════════════
# Response cleanup (module-level for use without class instance)
# ═════════════════════════════════════════════════════════════════════════════

def _clean_response(text: str) -> str:
    """Strip any trailing SKIP marker the LLM may have appended."""
    cleaned = re.sub(r"[\s\n]+SKIP\s*$", "", text.strip(), flags=re.IGNORECASE)
    return cleaned.strip()
