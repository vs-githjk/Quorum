"""
agent/intent.py — IntentDetector for Quorum

Classifies incoming transcript segments into actionable intent types using
fast keyword matching. Sets requires_llm=True only when keyword evidence is
ambiguous and an LLM is needed to confirm understanding.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Intent type constants ─────────────────────────────────────────────────────

QUESTION = "QUESTION"
ACTION_TASK = "ACTION_TASK"
ACTION_PR = "ACTION_PR"
ACTION_CHART = "ACTION_CHART"
DECISION = "DECISION"
TOPIC_MENTION = "TOPIC_MENTION"
NONE = "NONE"


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Intent:
    """
    The result of running IntentDetector.detect() on a transcript segment.

    Attributes:
        type:            One of the intent type constants defined above.
        confidence:      0.0–1.0. Keyword hits give 0.85; ambiguous topic
                         mentions give 0.55; no match gives 0.0.
        extracted_topic: The key term, feature name, or PR reference pulled
                         from the text. None if not applicable.
        raw_text:        The original transcript text that was classified.
        requires_llm:    True when the orchestrator should send this to the
                         LLM for deeper understanding before acting.
    """
    type: str
    confidence: float
    extracted_topic: str | None
    raw_text: str
    requires_llm: bool


# ── Keyword tables ────────────────────────────────────────────────────────────

# Each entry is a tuple of (phrase, is_prefix_match).
# is_prefix_match=False → phrase can appear anywhere in the lowercased text.
_QUESTION_PHRASES: list[str] = [
    "what did we",
    "do you remember",
    "what was",
    "can you find",
    "what's the status",
    "what is the status",
    "what happened to",
    "remind me",
    # Direct requests addressed to the bot
    "can you open",
    "can you show",
    "can you give me",
    "can you tell me",
    "can you pull",
    "can you check",
    "can you search",
    "could you open",
    "could you show",
    "could you give me",
    "could you tell me",
    "could you pull",
    "could you check",
    "could you find",
    "could you search",
]

_ACTION_TASK_PHRASES: list[str] = [
    "add that as a task",
    "create a ticket",
    "add to asana",
    "make a note",
    "action item",
    "create a task",
    "log that",
    "note that down",
    "add a task",
    "can you add",
    "could you add",
    "add that task",
]

_ACTION_PR_PHRASES: list[str] = [
    "pull up the pr",
    "show me the pr",
    "what's the pr for",
    "what is the pr for",
    "pull request for",
    "open the pr",
    "link the pr",
    "show the pull request",
]

_ACTION_CHART_PHRASES: list[str] = [
    "show me a chart",
    "can you visualize",
    "graph that",
    "show me the data",
    "plot that",
    "show a graph",
    "visualise",
    "visualize that",
]

# Phrases that introduce a topic by name — the noun phrase that follows is the topic.
# Lower confidence than proper-noun detection; useful when tech terms are lowercase.
_TOPIC_INTRODUCE_PHRASES: list[str] = [
    "let's talk about",
    "let us talk about",
    "let's discuss",
    "let us discuss",
    "talking about",
    "we need to discuss",
    "i want to discuss",
    "can we discuss",
]

_DECISION_PHRASES: list[str] = [
    "we'll go with",
    "we will go with",
    "let's do",
    "let us do",
    "agreed",
    "decided",
    "we're going to",
    "we are going to",
    "the plan is",
    "going with",
    "we've decided",
    "we have decided",
    "final decision",
    "final call",
]

# Quorum trigger phrases (mirrors mode.py — kept local to avoid circular import)
_Q_TRIGGERS: list[str] = [
    "q", "hey q",
    "hugh", "hey hugh",
    "cue", "hey cue",
    "que", "hey que",
    "ku", "hey ku",
    "kew", "hey kew",
    "queue", "hey queue",
    "aq", "hey aq",
]

# Proper-noun pattern: one or more capitalised words in a row, excluding
# sentence-start words (we strip leading caps from the first word of a sentence).
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

# Words to exclude from topic extraction (too generic)
_STOPWORDS: set[str] = {
    "The", "This", "That", "These", "Those", "There", "Their",
    "We", "I", "You", "He", "She", "It", "They",
    "A", "An", "And", "Or", "But", "So", "As", "At", "By",
    "In", "On", "To", "Of", "For", "With", "From", "Is",
    "Are", "Was", "Were", "Be", "Been", "Being", "Have",
    "Has", "Had", "Do", "Does", "Did", "Will", "Would",
    "Can", "Could", "Should", "May", "Might", "Must",
    "Shall", "Let", "Hey",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    # Social / filler words — never meaningful topics
    "Thank", "Thanks", "Okay", "Ok", "Sure", "Right", "Great",
    "Good", "Yes", "No", "Yeah", "Yep", "Nope", "Ah", "Oh",
    "Well", "Just", "Actually", "Basically", "Literally",
}


# ── IntentDetector ────────────────────────────────────────────────────────────

class IntentDetector:
    """
    Classifies transcript segments into intent types using keyword matching.

    Keyword matching is O(n·k) and runs in < 1 ms on typical segment lengths,
    making it safe to call on every incoming segment without API cost.
    LLM fallback is flagged via requires_llm=True but not triggered here —
    the orchestrator decides when to actually call the LLM.
    """

    def detect(self, segment) -> Intent:
        """
        Classify a TranscriptSegment and return an Intent.

        Detection priority (highest wins):
            1. ACTION_TASK
            2. ACTION_PR
            3. ACTION_CHART
            4. DECISION
            5. QUESTION
            6. TOPIC_MENTION
            7. NONE

        Args:
            segment: A TranscriptSegment (or any object with a .text attribute).

        Returns:
            Intent dataclass describing the detected intent.
        """
        text = segment.text
        lowered = text.lower()

        # ── Hard action intents (no LLM needed) ──────────────────────────────

        if self._matches(lowered, _ACTION_TASK_PHRASES):
            topic = self._first_noun_after_keyword(text, ["task", "ticket", "asana", "note"])
            logger.debug("Intent=ACTION_TASK text=%r topic=%r", text, topic)
            return Intent(
                type=ACTION_TASK,
                confidence=0.90,
                extracted_topic=topic,
                raw_text=text,
                requires_llm=False,
            )

        if self._matches(lowered, _ACTION_PR_PHRASES):
            topic = self._extract_pr_reference(text)
            logger.debug("Intent=ACTION_PR text=%r topic=%r", text, topic)
            return Intent(
                type=ACTION_PR,
                confidence=0.90,
                extracted_topic=topic,
                raw_text=text,
                requires_llm=False,
            )

        if self._matches(lowered, _ACTION_CHART_PHRASES):
            topic = self._first_noun_after_keyword(text, ["chart", "graph", "data", "plot"])
            logger.debug("Intent=ACTION_CHART text=%r topic=%r", text, topic)
            return Intent(
                type=ACTION_CHART,
                confidence=0.90,
                extracted_topic=topic,
                raw_text=text,
                requires_llm=False,
            )

        # ── Decision (no LLM needed) ──────────────────────────────────────────

        if self._matches(lowered, _DECISION_PHRASES):
            topic = self._extract_decision_subject(text)
            logger.debug("Intent=DECISION text=%r topic=%r", text, topic)
            return Intent(
                type=DECISION,
                confidence=0.85,
                extracted_topic=topic,
                raw_text=text,
                requires_llm=False,
            )

        # ── Question (may need LLM for ambiguous phrasing) ───────────────────

        if self._matches(lowered, _QUESTION_PHRASES):
            topic = self.extract_topics(text)
            extracted = topic[0] if topic else None
            # Ambiguous questions benefit from LLM rephrasing before search
            needs_llm = extracted is None
            logger.debug("Intent=QUESTION text=%r topic=%r requires_llm=%s", text, extracted, needs_llm)
            return Intent(
                type=QUESTION,
                confidence=0.85,
                extracted_topic=extracted,
                raw_text=text,
                requires_llm=needs_llm,
            )

        # ── Topic introduction ("let's talk about X") ────────────────────────
        # Catches cases where the topic is lowercase (e.g. "authentication feature")
        # and no proper noun is present.

        if self._matches(lowered, _TOPIC_INTRODUCE_PHRASES):
            topic = self._noun_after_introduce_phrase(text)
            logger.debug("Intent=TOPIC_MENTION (introduce) text=%r topic=%r", text, topic)
            return Intent(
                type=TOPIC_MENTION,
                confidence=0.70,
                extracted_topic=topic,
                raw_text=text,
                requires_llm=True,
            )

        # ── Topic mention (capitalized proper nouns) ──────────────────────────

        topics = self.extract_topics(text)
        if topics:
            # Use LLM to decide if this topic is actually worth surfacing context for
            logger.debug("Intent=TOPIC_MENTION text=%r topics=%r", text, topics)
            return Intent(
                type=TOPIC_MENTION,
                confidence=0.55,
                extracted_topic=topics[0],
                raw_text=text,
                requires_llm=True,
            )

        # ── Nothing actionable ────────────────────────────────────────────────

        logger.debug("Intent=NONE text=%r", text)
        return Intent(
            type=NONE,
            confidence=0.0,
            extracted_topic=None,
            raw_text=text,
            requires_llm=False,
        )

    def extract_topics(self, text: str) -> list[str]:
        """
        Pull named entities (capitalized proper nouns) from text.

        Strips generic stopwords and single-character tokens. Returns unique
        topics in order of first appearance.

        Args:
            text: Raw transcript text.

        Returns:
            Ordered list of unique proper-noun strings worth searching for.
        """
        # Strip the very first word of the text from caps consideration
        # (sentences start with a capital regardless of proper-noun status)
        stripped = re.sub(r"^\s*\S+\s*", "", text)
        matches = _PROPER_NOUN_RE.findall(stripped)

        seen: set[str] = set()
        topics: list[str] = []
        for m in matches:
            if m in _STOPWORDS:
                continue
            if len(m) <= 1:
                continue
            if m not in seen:
                seen.add(m)
                topics.append(m)
        return topics

    def is_addressed_to_q(self, text: str) -> bool:
        """
        Return True if the text explicitly addresses Quorum by name.

        Args:
            text: Raw transcript text.

        Returns:
            True if 'quorum' or 'hey quorum' appears (case-insensitive).
        """
        lowered = text.lower()
        return any(trigger in lowered for trigger in _Q_TRIGGERS)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _noun_after_introduce_phrase(text: str) -> str | None:
        """
        Extract the noun phrase that follows a topic-introduction trigger.

        e.g. "Let's talk about the new authentication feature"
              → "the new authentication feature"

        Args:
            text: Original transcript text.

        Returns:
            The phrase following the trigger keyword, up to 6 words, or None.
        """
        lowered = text.lower()
        for phrase in _TOPIC_INTRODUCE_PHRASES:
            idx = lowered.find(phrase)
            if idx == -1:
                continue
            after = text[idx + len(phrase):].strip().lstrip(".,!? ")
            snippet = re.split(r"[.,!?]", after)[0]
            words = snippet.split()[:6]
            result = " ".join(words).strip()
            return result if result else None
        return None

    @staticmethod
    def _matches(lowered_text: str, phrases: list[str]) -> bool:
        """Return True if any phrase from the list appears in lowered_text."""
        return any(phrase in lowered_text for phrase in phrases)

    @staticmethod
    def _first_noun_after_keyword(text: str, keywords: list[str]) -> str | None:
        """
        Find the first keyword in text and return the noun phrase that follows it.

        Args:
            text:     Original (mixed-case) transcript text.
            keywords: Lowercase keywords to scan for.

        Returns:
            The noun phrase after the keyword, or None if nothing useful follows.
        """
        lowered = text.lower()
        for kw in keywords:
            idx = lowered.find(kw)
            if idx == -1:
                continue
            after = text[idx + len(kw):].strip().lstrip(".,!? ")
            # Return up to the first punctuation or 5 words
            snippet = re.split(r"[.,!?]", after)[0]
            words = snippet.split()[:5]
            result = " ".join(words).strip()
            return result if result else None
        return None

    @staticmethod
    def _extract_pr_reference(text: str) -> str | None:
        """
        Extract a PR number (#123) or branch/feature name from text.

        Args:
            text: Raw transcript text.

        Returns:
            PR reference string or None.
        """
        # Look for explicit PR numbers first
        match = re.search(r"#(\d+)", text)
        if match:
            return f"PR #{match.group(1)}"

        # Fall back to the noun phrase after "PR" or "pull request"
        match = re.search(
            r"(?:PR|pull request)\s+(?:for\s+)?([A-Za-z0-9_\-\s]{2,30})",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    @staticmethod
    def _extract_decision_subject(text: str) -> str | None:
        """
        Extract what the decision is about from a DECISION-classified segment.

        Args:
            text: Raw transcript text.

        Returns:
            A short subject string, or None.
        """
        # Grab what comes after decision trigger phrases
        for phrase in ["we'll go with", "we will go with", "going with",
                        "let's do", "let us do", "the plan is"]:
            idx = text.lower().find(phrase)
            if idx != -1:
                after = text[idx + len(phrase):].strip().lstrip(".,!? ")
                words = after.split()[:6]
                result = " ".join(words).rstrip(".,!?").strip()
                return result if result else None
        return None


# ── Standalone test ───────────────────────────────────────────────────────────

def test() -> None:
    """
    Run 10 sample transcript lines through IntentDetector and print results.

    Usage:
        python3 -m agent.intent
    """
    import types

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    def seg(text: str):
        """Build a minimal mock TranscriptSegment."""
        s = types.SimpleNamespace()
        s.text = text
        s.speaker = "Speaker 0"
        s.timestamp = 0.0
        s.is_final = True
        s.meeting_id = "test-001"
        return s

    samples = [
        # (text, expected_intent_type)
        ("What did we decide about the API rate limits?",             QUESTION),
        ("Add that as a task for the backend team",                   ACTION_TASK),
        ("Can you pull up the PR for authentication?",                ACTION_PR),
        ("Show me a chart of our sprint velocity",                    ACTION_CHART),
        ("We'll go with OAuth 2.0, everyone agreed",                  DECISION),
        ("Let's talk about the new Stripe integration",               TOPIC_MENTION),
        ("Hey Quorum, what's the status of the migration?",           QUESTION),
        ("Agreed, we're going with the microservices approach",       DECISION),
        ("Can you find the Notion doc for onboarding?",               QUESTION),
        ("The weather is nice today",                                 NONE),
    ]

    detector = IntentDetector()

    print("\n=== IntentDetector test ===\n")
    print(f"{'Text':<52} {'Expected':<15} {'Got':<15} {'Topic':<20} {'LLM?'}")
    print("-" * 115)

    passed = 0
    for text, expected in samples:
        intent = detector.detect(seg(text))
        ok = "PASS" if intent.type == expected else "FAIL"
        if ok == "PASS":
            passed += 1
        topic = (intent.extracted_topic or "—")[:20]
        print(
            f"[{ok}] {text[:50]:<52} {expected:<15} {intent.type:<15} "
            f"{topic:<20} requires_llm={intent.requires_llm}"
        )

    print(f"\n{passed}/{len(samples)} passed\n")


if __name__ == "__main__":
    test()
