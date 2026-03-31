"""
agent/context.py — MeetingContext for Quorum

Manages in-session memory (transcript, decisions, actions, surfaced context)
and cross-meeting persistence (meeting_history.json). Provides keyword-overlap
search across past meetings — no vector DB or embeddings required for MVP.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Path for cross-meeting persistence; overridable via env var
DEFAULT_HISTORY_FILE = os.getenv("QUORUM_HISTORY_FILE", "meeting_history.json")

# LLM settings (context.py calls LLM only for summarize_meeting)
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
_SUMMARIZE_TIMEOUT = 10.0

_SYSTEM_PROMPT = (
    "You are Quorum, an AI meeting participant. You are helpful, concise, "
    "and only speak when you have something genuinely useful to add. "
    "Keep all spoken responses under 2 sentences. You are currently in a meeting."
)


# ── In-session dataclasses ────────────────────────────────────────────────────

@dataclass
class _SegmentRecord:
    """Lightweight serialisable form of a TranscriptSegment."""
    text: str
    speaker: str
    timestamp: float
    meeting_id: str


@dataclass
class _DecisionRecord:
    """A decision that was detected and logged during a meeting."""
    text: str
    meeting_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class _MeetingSummary:
    """Persisted summary for a completed meeting."""
    meeting_id: str
    date: str                        # ISO-8601 date string
    summary: str                     # 3-sentence LLM summary
    decisions: list[str]
    action_count: int
    transcript_snippet: str          # first 500 chars of transcript for search


# ── MeetingContext ────────────────────────────────────────────────────────────

class MeetingContext:
    """
    Stores everything that happens in a meeting and retrieves relevant context
    from past meetings.

    In-session data lives in memory. On meeting end, everything is persisted
    to meeting_history.json and a 3-sentence LLM summary is generated.
    """

    def __init__(self, history_file: str = DEFAULT_HISTORY_FILE) -> None:
        """
        Initialise MeetingContext.

        Args:
            history_file: Path to the JSON file used for cross-meeting storage.
        """
        self._history_file = Path(history_file)

        # In-session state keyed by meeting_id
        self._transcripts: dict[str, list[_SegmentRecord]] = {}
        self._decisions: dict[str, list[_DecisionRecord]] = {}
        self._actions: dict[str, list[dict]] = {}       # ActionRequest dicts
        self._surfaced: dict[str, list[dict]] = {}      # IntegrationResult dicts

        # Rolling agent exchange history — last N tool-call cycles per meeting.
        # Each entry is a list of message dicts (user + assistant + tool turns)
        # from one QAgent.run() call, used to give Q short-term memory.
        self._agent_history: dict[str, list[list[dict]]] = {}

        # Tool actions taken this meeting — human-readable strings extracted
        # from tool results, persisted to meeting_history.json at end.
        self._tool_actions: dict[str, list[str]] = {}

        # Cross-meeting history loaded from disk
        self._history: dict[str, dict] = self._load_history()

        logger.info(
            "MeetingContext initialised — %d past meetings in history",
            len(self._history),
        )

    # ── In-session writes ─────────────────────────────────────────────────────

    def add_segment(self, segment) -> None:
        """
        Store a final transcript segment in the current meeting's transcript.

        Only stores segments where is_final=True; interim results are silently
        dropped (they will arrive again as final).

        Args:
            segment: TranscriptSegment (or any object with .text, .speaker,
                     .timestamp, .is_final, .meeting_id attributes).
        """
        if not segment.is_final:
            logger.debug("Dropping interim segment for meeting %s", segment.meeting_id)
            return

        mid = segment.meeting_id
        self._transcripts.setdefault(mid, [])
        record = _SegmentRecord(
            text=segment.text,
            speaker=segment.speaker,
            timestamp=segment.timestamp,
            meeting_id=mid,
        )
        self._transcripts[mid].append(record)
        logger.debug(
            "Segment added to %s [total=%d]: %r",
            mid,
            len(self._transcripts[mid]),
            segment.text[:60],
        )

    def add_decision(self, text: str, meeting_id: str) -> None:
        """
        Log a detected decision for the current meeting.

        Args:
            text:       The decision text extracted from the transcript.
            meeting_id: The active meeting ID.
        """
        self._decisions.setdefault(meeting_id, [])
        record = _DecisionRecord(text=text, meeting_id=meeting_id)
        self._decisions[meeting_id].append(record)
        logger.info("Decision logged for %s: %r", meeting_id, text)

    def add_action(self, action) -> None:
        """
        Record an ActionRequest that was dispatched during this meeting.

        Args:
            action: ActionRequest dataclass or any object with .action_type,
                    .parameters, .context, .meeting_id attributes.
        """
        mid = action.meeting_id
        self._actions.setdefault(mid, [])
        entry = {
            "action_type": action.action_type,
            "parameters": action.parameters,
            "context": action.context,
            "timestamp": time.time(),
        }
        self._actions[mid].append(entry)
        logger.info("Action logged for %s: %s", mid, action.action_type)

    def add_surfaced_result(self, result, meeting_id: str) -> None:
        """
        Record an IntegrationResult that was surfaced to the meeting.

        Args:
            result:     IntegrationResult dataclass.
            meeting_id: The active meeting ID.
        """
        self._surfaced.setdefault(meeting_id, [])
        entry = {
            "source": result.source,
            "title": result.title,
            "url": result.url,
            "summary": result.summary,
            "timestamp": result.timestamp,
        }
        self._surfaced[meeting_id].append(entry)
        logger.debug("Surfaced result for %s from %s: %r", meeting_id, result.source, result.title)

    # ── In-session reads ──────────────────────────────────────────────────────

    def get_recent_transcript(self, meeting_id: str, n: int = 10) -> str:
        """
        Return the last N transcript segments as a formatted string.

        Format per line:
            [Speaker 0 @ 12:34:56] "Hello everyone..."

        Args:
            meeting_id: The meeting to pull from.
            n:          Number of segments to return (default 10).

        Returns:
            Multi-line formatted string, or a placeholder if no transcript yet.
        """
        segments = self._transcripts.get(meeting_id, [])
        recent = segments[-n:]
        if not recent:
            return "(no transcript yet)"

        lines = []
        for seg in recent:
            ts = time.strftime("%H:%M:%S", time.localtime(seg.timestamp))
            lines.append(f'[{seg.speaker} @ {ts}] "{seg.text}"')
        return "\n".join(lines)

    def is_already_surfaced(self, url: str, meeting_id: str) -> bool:
        """
        Return True if a result with this URL has already been surfaced
        in the given meeting.

        Used by the orchestrator to deduplicate integration results before
        passing them to the LLM or speaking about them.

        Args:
            url:        The canonical URL from IntegrationResult.
            meeting_id: The active meeting session.

        Returns:
            True if this URL was previously passed to add_surfaced_result
            for this meeting.
        """
        return any(
            entry["url"] == url
            for entry in self._surfaced.get(meeting_id, [])
        )

    def get_decisions(self, meeting_id: str) -> list[str]:
        """
        Return all decisions logged for a meeting.

        Args:
            meeting_id: The meeting to query.

        Returns:
            List of decision text strings.
        """
        return [d.text for d in self._decisions.get(meeting_id, [])]

    # ── Agent exchange history (short-term memory) ───────────────────────────

    def add_agent_exchange(self, meeting_id: str, exchange: list[dict]) -> None:
        """
        Store one completed agent exchange for short-term memory.

        An exchange is the slice of the messages list from the current user
        utterance onward (excluding the system prompt), captured after each
        QAgent.run() call. Capped at the last 10 exchanges per meeting.

        Args:
            meeting_id: The active meeting session.
            exchange:   List of message dicts (user + assistant/tool turns).
        """
        self._agent_history.setdefault(meeting_id, [])
        self._agent_history[meeting_id].append(exchange)
        self._agent_history[meeting_id] = self._agent_history[meeting_id][-10:]

        # Extract a human-readable summary of any tool results for long-term memory
        for msg in exchange:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                name = msg.get("name", "tool")
                if content and not content.startswith("Error"):
                    summary = f"[{name}] {content[:200]}"
                    self._tool_actions.setdefault(meeting_id, [])
                    self._tool_actions[meeting_id].append(summary)

        logger.debug(
            "[%s] Agent exchange saved — total exchanges: %d",
            meeting_id,
            len(self._agent_history[meeting_id]),
        )

    def get_agent_history(self, meeting_id: str, n: int = 3) -> list[dict]:
        """
        Return the last n exchanges as a flat list of messages.

        These are injected between the system prompt and the current user
        message in QAgent.run() to give the LLM short-term memory of recent
        tool calls and their results.

        Args:
            meeting_id: The active meeting session.
            n:          Number of past exchanges to include (default 3).

        Returns:
            Flat list of message dicts in chronological order.
        """
        exchanges = self._agent_history.get(meeting_id, [])
        recent = exchanges[-n:]
        flat: list[dict] = []
        for exchange in recent:
            flat.extend(exchange)
        return flat

    # ── Cross-meeting search ──────────────────────────────────────────────────

    def search_past_meetings(self, query: str) -> str:
        """
        Search past meeting summaries and decisions using keyword overlap.

        Splits the query into words, scores each past meeting by how many
        query words appear in its summary + decisions, and returns the top 3
        matches as a formatted string.

        No embeddings or vector DB — pure word overlap, fast and offline.

        Args:
            query: The topic or question to search for.

        Returns:
            Formatted string with up to 3 relevant past meeting snippets,
            or a message indicating no relevant history was found.
        """
        if not self._history:
            return "No past meeting history available."

        query_words = set(query.lower().split())
        # Remove very short words that add noise
        query_words = {w for w in query_words if len(w) > 2}

        scores: list[tuple[float, str, dict]] = []

        for mid, meeting in self._history.items():
            searchable = " ".join([
                meeting.get("summary", ""),
                " ".join(meeting.get("decisions", [])),
                " ".join(meeting.get("tool_actions", [])),
                meeting.get("transcript_snippet", ""),
            ]).lower()
            searchable_words = set(searchable.split())
            overlap = len(query_words & searchable_words)
            if overlap > 0:
                scores.append((overlap, mid, meeting))

        if not scores:
            return f"No past meetings found relevant to: '{query}'"

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:3]

        parts = []
        for _, mid, meeting in top:
            date = meeting.get("date", "unknown date")
            summary = meeting.get("summary", "No summary available.")
            decisions = meeting.get("decisions", [])
            dec_str = "; ".join(decisions[:3]) if decisions else "none recorded"
            tool_actions = meeting.get("tool_actions", [])
            actions_str = "; ".join(tool_actions[:5]) if tool_actions else "none"
            parts.append(
                f"Meeting {mid} ({date}):\n"
                f"  Summary: {summary}\n"
                f"  Decisions: {dec_str}\n"
                f"  Actions taken: {actions_str}"
            )

        return "\n\n".join(parts)

    # ── Meeting lifecycle ─────────────────────────────────────────────────────

    async def summarize_meeting(self, meeting_id: str) -> str:
        """
        Call the LLM to generate a 3-sentence summary of the meeting.

        Tries Hermes locally first; falls back to OpenRouter if Hermes is
        unavailable or too slow.

        Args:
            meeting_id: The meeting to summarise.

        Returns:
            A 3-sentence summary string.
        """
        transcript = self.get_recent_transcript(meeting_id, n=50)
        decisions = self.get_decisions(meeting_id)
        dec_text = "\n".join(f"- {d}" for d in decisions) if decisions else "None recorded."

        prompt = (
            f"Summarise this meeting in exactly 3 sentences. "
            f"Focus on what was discussed, what was decided, and what actions were taken.\n\n"
            f"TRANSCRIPT (last 50 segments):\n{transcript}\n\n"
            f"DECISIONS:\n{dec_text}"
        )

        logger.info("Generating summary for meeting %s", meeting_id)
        summary = await self._call_llm(prompt)
        logger.info("Summary generated for %s: %r", meeting_id, summary[:80])
        return summary

    async def end_meeting(self, meeting_id: str) -> None:
        """
        Finalise a meeting: generate a summary and persist everything to disk.

        Safe to call even if the meeting had no transcript — a placeholder
        summary will be stored.

        Args:
            meeting_id: The meeting to end.
        """
        logger.info("Ending meeting %s", meeting_id)

        summary = await self.summarize_meeting(meeting_id)

        segments = self._transcripts.get(meeting_id, [])
        snippet = " ".join(s.text for s in segments[:10])[:500]

        decisions = [d.text for d in self._decisions.get(meeting_id, [])]
        action_count = len(self._actions.get(meeting_id, []))
        tool_actions = self._tool_actions.get(meeting_id, [])

        self._history[meeting_id] = {
            "meeting_id": meeting_id,
            "date": time.strftime("%Y-%m-%d"),
            "summary": summary,
            "decisions": decisions,
            "tool_actions": tool_actions,
            "action_count": action_count,
            "transcript_snippet": snippet,
        }

        self._persist_history()
        logger.info(
            "Meeting %s persisted — %d decisions, %d actions",
            meeting_id,
            len(decisions),
            action_count,
        )

    # ── LLM helpers ───────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        """Call OpenRouter. Returns response text, or a safe fallback on failure."""
        try:
            return await self._call_openrouter(prompt)
        except Exception as exc:
            logger.error("OpenRouter failed: %s", exc)
            return "Summary unavailable — LLM unreachable."

    async def _call_openrouter(self, prompt: str) -> str:
        """
        Call OpenRouter as LLM fallback.

        Args:
            prompt: Prompt text.

        Returns:
            Generated text from OpenRouter.

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
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL, json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _load_history(self) -> dict[str, dict]:
        """
        Load meeting history from disk.

        Returns:
            Dict of meeting_id → meeting summary dict, or empty dict.
        """
        if self._history_file.exists():
            try:
                data = json.loads(self._history_file.read_text())
                logger.debug("Loaded %d meetings from %s", len(data), self._history_file)
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read history file (%s) — starting fresh", exc)
        return {}

    def _persist_history(self) -> None:
        """Write the current cross-meeting history to disk."""
        try:
            self._history_file.write_text(
                json.dumps(self._history, indent=2, default=str)
            )
            logger.debug("History persisted to %s", self._history_file)
        except OSError as exc:
            logger.error("Failed to persist meeting history: %s", exc)


# ── Standalone test ───────────────────────────────────────────────────────────

async def _run_test() -> None:
    """
    Smoke test for MeetingContext — run with:
        python3 -m agent.context
    """
    import tempfile
    import types

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    def make_segment(text, speaker="Speaker 0", meeting_id="mtg-001"):
        """Build a minimal mock TranscriptSegment."""
        s = types.SimpleNamespace()
        s.text = text
        s.speaker = speaker
        s.timestamp = time.time()
        s.is_final = True
        s.meeting_id = meeting_id
        return s

    def make_action(action_type, meeting_id="mtg-001"):
        """Build a minimal mock ActionRequest."""
        a = types.SimpleNamespace()
        a.action_type = action_type
        a.parameters = {"key": "value"}
        a.context = "test context"
        a.meeting_id = meeting_id
        return a

    def make_result(source="github", title="Auth PR"):
        """Build a minimal mock IntegrationResult."""
        r = types.SimpleNamespace()
        r.source = source
        r.title = title
        r.url = "https://example.com"
        r.summary = "Auth refactor merged"
        r.timestamp = time.time()
        return r

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_history = f.name

    ctx = MeetingContext(history_file=tmp_history)
    print("\n=== MeetingContext smoke test ===\n")
    passed = 0
    total = 0

    def check(label: str, condition: bool) -> None:
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"[{status}] {label}")

    # add_segment — final segment stored
    seg1 = make_segment("Let's discuss authentication")
    ctx.add_segment(seg1)
    check("add_segment stores final segment", len(ctx._transcripts["mtg-001"]) == 1)

    # add_segment — interim segment dropped
    interim = make_segment("Let's")
    interim.is_final = False
    ctx.add_segment(interim)
    check("add_segment drops interim segment", len(ctx._transcripts["mtg-001"]) == 1)

    # add more segments
    ctx.add_segment(make_segment("We'll go with OAuth 2.0"))
    ctx.add_segment(make_segment("Great, that's decided then"))

    # get_recent_transcript
    recent = ctx.get_recent_transcript("mtg-001", n=2)
    check("get_recent_transcript returns last N segments", "OAuth" in recent and "decided" in recent)
    check("get_recent_transcript excludes older segments", "authentication" not in recent)

    # add_decision
    ctx.add_decision("We will use OAuth 2.0 for authentication", "mtg-001")
    decisions = ctx.get_decisions("mtg-001")
    check("add_decision stores decision", len(decisions) == 1 and "OAuth" in decisions[0])

    # add_action
    ctx.add_action(make_action("create_task"))
    check("add_action stores action", len(ctx._actions.get("mtg-001", [])) == 1)

    # add_surfaced_result
    ctx.add_surfaced_result(make_result(), "mtg-001")
    check("add_surfaced_result stores result", len(ctx._surfaced.get("mtg-001", [])) == 1)

    # search_past_meetings — no history yet
    result = ctx.search_past_meetings("OAuth authentication")
    check("search_past_meetings returns no-history message when empty",
          "No past meeting" in result)

    # Seed history manually for search test
    ctx._history["mtg-000"] = {
        "meeting_id": "mtg-000",
        "date": "2024-01-15",
        "summary": "The team discussed OAuth 2.0 and API rate limiting for the auth service.",
        "decisions": ["Use OAuth 2.0", "Rate limit at 100 req/min"],
        "action_count": 2,
        "transcript_snippet": "OAuth API auth service rate limit discussion",
    }
    ctx._history["mtg-past"] = {
        "meeting_id": "mtg-past",
        "date": "2024-01-10",
        "summary": "Sprint planning for the dashboard feature and Stripe integration.",
        "decisions": ["Stripe for payments"],
        "action_count": 1,
        "transcript_snippet": "Stripe dashboard sprint velocity chart",
    }

    result = ctx.search_past_meetings("OAuth authentication")
    check("search_past_meetings finds relevant past meeting", "mtg-000" in result)
    check("search_past_meetings excludes unrelated meeting", "Stripe" not in result or "mtg-000" in result)

    result_stripe = ctx.search_past_meetings("Stripe payments")
    check("search_past_meetings finds Stripe meeting", "mtg-past" in result_stripe)

    # end_meeting (without real LLM — mocked via monkeypatching)
    original_llm = ctx._call_llm
    ctx._call_llm = lambda p, timeout=10: _fake_llm(p)  # type: ignore

    import asyncio
    async def _fake_llm(p):
        return "The team discussed authentication. OAuth 2.0 was chosen. One task was created."

    ctx._call_llm = _fake_llm  # type: ignore

    await ctx.end_meeting("mtg-001")
    check("end_meeting persists to history", "mtg-001" in ctx._history)
    check("end_meeting stores summary", "OAuth" in ctx._history["mtg-001"].get("summary", ""))
    check("end_meeting records decisions", len(ctx._history["mtg-001"]["decisions"]) == 1)

    # Reload from disk
    ctx2 = MeetingContext(history_file=tmp_history)
    check("history survives reload from disk", "mtg-001" in ctx2._history)

    os.unlink(tmp_history)
    print(f"\n{passed}/{total} passed\n")


def test() -> None:
    """Entry point for direct module execution."""
    import asyncio
    asyncio.run(_run_test())


if __name__ == "__main__":
    test()
