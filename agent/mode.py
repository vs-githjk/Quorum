"""
agent/mode.py — ModeManager for Q

Controls whether Q responds proactively (ACTIVE) or only when
explicitly addressed (ON_DEMAND). Mode is persisted to mode_state.json
so it survives process restarts.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ACTIVE = "active"
ON_DEMAND = "on_demand"
VALID_MODES = {ACTIVE, ON_DEMAND}

# Trigger phrases that count as addressing Q directly.
# Includes common Deepgram phonetic mishearings of the letter "Q".
Q_TRIGGERS = [
    "q", "hey q",
    # Deepgram mishearings observed in the wild
    "hugh", "hey hugh",
    "cue", "hey cue",
    "que", "hey que",
    "ku", "hey ku",
    "kew", "hey kew",
    "queue", "hey queue",
    "aq", "hey aq",
]

# Default path for persisted mode; can be overridden via env var
DEFAULT_STATE_FILE = os.getenv("QUORUM_MODE_STATE_FILE", "mode_state.json")


class ModeManager:
    """
    Manages the operational mode for the Quorum agent.

    ACTIVE    — Quorum speaks proactively whenever it has relevant context.
    ON_DEMAND — Quorum only speaks when "Q" or "hey quorum" is detected
                in the transcript (case-insensitive).

    Mode is written to a JSON file on every change so restarts pick up
    where they left off. The default on first run is ON_DEMAND to prevent
    Quorum from speaking unexpectedly at the start of a meeting.
    """

    def __init__(self, state_file: str = DEFAULT_STATE_FILE) -> None:
        """
        Initialise ModeManager, loading persisted state if available.

        Args:
            state_file: Path to the JSON file used for persistence.
        """
        self._state_file = Path(state_file)
        self._mode = self._load_mode()
        logger.info("ModeManager initialised — current mode: %s", self._mode)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_mode(self) -> str:
        """Return the current mode string ('active' or 'on_demand')."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """
        Set the operational mode and persist it to disk.

        Args:
            mode: Must be 'active' or 'on_demand'.

        Raises:
            ValueError: If mode is not one of the valid values.
        """
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {VALID_MODES}."
            )
        previous = self._mode
        self._mode = mode
        self._persist_mode()
        logger.info("Mode changed: %s → %s", previous, mode)

    def should_respond(
        self, transcript_text: str, has_relevant_context: bool
    ) -> bool:
        """
        Decide whether Quorum should respond to this transcript segment.

        In ACTIVE mode:    returns True if has_relevant_context is True.
        In ON_DEMAND mode: returns True only when 'quorum' or 'hey quorum'
                           appears in transcript_text (case-insensitive).

        Args:
            transcript_text:     The spoken text from the current segment.
            has_relevant_context: Whether the agent found something worth saying.

        Returns:
            True if Quorum should speak, False otherwise.
        """
        if self._mode == ACTIVE:
            result = has_relevant_context
            logger.debug(
                "ACTIVE mode — has_relevant_context=%s → respond=%s",
                has_relevant_context,
                result,
            )
            return result

        # ON_DEMAND: only act when explicitly addressed
        addressed = self.is_addressed_to_q(transcript_text)
        logger.debug(
            "ON_DEMAND mode — addressed=%s → respond=%s", addressed, addressed
        )
        return addressed

    def toggle(self) -> str:
        """
        Flip between ACTIVE and ON_DEMAND, persist, and return the new mode.

        Returns:
            The new mode string after toggling.
        """
        new_mode = ON_DEMAND if self._mode == ACTIVE else ACTIVE
        self.set_mode(new_mode)
        logger.info("Mode toggled to: %s", new_mode)
        return new_mode

    # ── Helpers ──────────────────────────────────────────────────────────────

    def is_addressed_to_q(self, text: str) -> bool:
        """
        Return True if any Quorum trigger phrase appears in text.

        Strips punctuation before matching so "Hey, Q." and "Hey Q" both hit.

        Args:
            text: Raw transcript text to check.

        Returns:
            True if 'quorum' or 'hey quorum' is found (case-insensitive).
        """
        import re as _re
        # Normalise: strip punctuation, collapse whitespace
        normalised = _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", " ", text.lower())).strip()
        words = normalised.split()
        # Use word-boundary matching — "q" must be a standalone word, not inside
        # "request", "question", "quite" etc.
        for trigger in Q_TRIGGERS:
            trigger_words = trigger.split()
            # Sliding window match
            for i in range(len(words) - len(trigger_words) + 1):
                if words[i:i + len(trigger_words)] == trigger_words:
                    return True
        return False

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_mode(self) -> str:
        """
        Load mode from the state file, falling back to ON_DEMAND.

        Returns:
            The persisted mode string, or ON_DEMAND if the file is absent
            or contains an invalid value.
        """
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                mode = data.get("mode", ON_DEMAND)
                if mode not in VALID_MODES:
                    logger.warning(
                        "Invalid mode '%s' in state file — defaulting to %s",
                        mode,
                        ON_DEMAND,
                    )
                    return ON_DEMAND
                logger.debug("Loaded persisted mode: %s", mode)
                return mode
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read state file (%s) — defaulting to %s",
                    exc,
                    ON_DEMAND,
                )
        return ON_DEMAND

    def _persist_mode(self) -> None:
        """Write the current mode to the state file."""
        try:
            self._state_file.write_text(json.dumps({"mode": self._mode}, indent=2))
            logger.debug("Persisted mode '%s' to %s", self._mode, self._state_file)
        except OSError as exc:
            logger.error("Failed to persist mode state: %s", exc)


# ── Standalone smoke test ─────────────────────────────────────────────────────

def test() -> None:
    """
    Quick smoke test for ModeManager — run directly with:
        python -m agent.mode
    """
    import tempfile

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name

    print("\n=== ModeManager smoke test ===\n")

    mgr = ModeManager(state_file=tmp)
    assert mgr.get_mode() == ON_DEMAND, "Default should be ON_DEMAND"
    print(f"[PASS] Default mode: {mgr.get_mode()}")

    # ON_DEMAND — not addressed → should NOT respond
    result = mgr.should_respond("Let's talk about the sprint", True)
    assert result is False
    print(f"[PASS] ON_DEMAND, not addressed, has_context=True → {result}")

    # ON_DEMAND — addressed → should respond
    result = mgr.should_respond("Hey Quorum, what did we decide?", True)
    assert result is True
    print(f"[PASS] ON_DEMAND, addressed → {result}")

    # ON_DEMAND — 'quorum' anywhere in text → should respond
    result = mgr.should_respond("Can quorum pull up the PR?", False)
    assert result is True
    print(f"[PASS] ON_DEMAND, 'quorum' mid-sentence, has_context=False → {result}")

    # Toggle to ACTIVE
    new_mode = mgr.toggle()
    assert new_mode == ACTIVE
    print(f"[PASS] toggle() → {new_mode}")

    # ACTIVE — has_context=False → should NOT respond
    result = mgr.should_respond("Nothing relevant here", False)
    assert result is False
    print(f"[PASS] ACTIVE, has_context=False → {result}")

    # ACTIVE — has_context=True → should respond
    result = mgr.should_respond("Nothing relevant here", True)
    assert result is True
    print(f"[PASS] ACTIVE, has_context=True → {result}")

    # Persistence: new instance should reload ACTIVE
    mgr2 = ModeManager(state_file=tmp)
    assert mgr2.get_mode() == ACTIVE
    print(f"[PASS] Reloaded from disk → {mgr2.get_mode()}")

    # set_mode with invalid value raises
    try:
        mgr.set_mode("turbo")
        print("[FAIL] Should have raised ValueError")
    except ValueError as exc:
        print(f"[PASS] Invalid mode rejected: {exc}")

    os.unlink(tmp)
    print("\n=== All tests passed ===\n")


if __name__ == "__main__":
    test()
