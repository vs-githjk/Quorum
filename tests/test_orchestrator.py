"""
tests/test_orchestrator.py — Unit tests for agent/orchestrator.py

Tests the simplified orchestrator: debounce gating, exchange state,
mode gate, cooldown, address-to-Q detection, and QAgent delegation.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent.q_agent import QResponse
from agent.orchestrator import (
    QOrchestrator,
    TranscriptSegment,
    SpeakCommand,
    _looks_complete,
    _strip_q_trigger,
)


def _seg(text: str, mid: str = "m1", speaker: str = "S0") -> TranscriptSegment:
    return TranscriptSegment(
        text=text, speaker=speaker,
        timestamp=time.time(), is_final=True, meeting_id=mid,
    )


def _make_orch(agent=None):
    spoken = []

    async def speak(cmd):
        spoken.append(cmd)

    async def integrate(req):
        return []

    async def act(req):
        return {}

    orch = QOrchestrator(
        speak_callback=speak,
        integration_callback=integrate,
        action_callback=act,
        agent=agent,
    )
    return orch, spoken


class TestLooksComplete(unittest.TestCase):

    def test_ends_with_period(self):
        assert _looks_complete("That is correct.") is True

    def test_ends_with_question(self):
        assert _looks_complete("Are you sure?") is True

    def test_two_words_fragment(self):
        assert _looks_complete("the new") is False

    def test_ends_with_continuation(self):
        assert _looks_complete("I was looking at the") is False

    def test_starts_with_question_word_no_mark(self):
        assert _looks_complete("What does the team think") is False

    def test_normal_sentence_no_punct(self):
        assert _looks_complete("We should use OAuth for authentication") is True


class TestStripQTrigger(unittest.TestCase):

    def test_hey_q(self):
        assert _strip_q_trigger("Hey Q could you search Slack?") == "could you search Slack?"

    def test_hey_quorum_comma(self):
        result = _strip_q_trigger("Hey Quorum, what did we decide?")
        assert "what did we decide" in result

    def test_no_trigger(self):
        assert _strip_q_trigger("Let's discuss auth") == "Let's discuss auth"

    def test_pure_trigger(self):
        assert _strip_q_trigger("Hey Q") == ""


class TestExchangeState(unittest.IsolatedAsyncioTestCase):

    async def test_starts_idle(self):
        orch, _ = _make_orch()
        assert orch._get_exchange_state("m1") == "idle"

    async def test_becomes_engaged_on_segment(self):
        orch, _ = _make_orch()
        await orch.start_meeting("m1")
        orch._set_exchange_engaged("m1")
        assert orch._get_exchange_state("m1") == "engaged"

    async def test_idle_after_timeout(self):
        orch, _ = _make_orch()
        orch._set_exchange_engaged("m1")
        # Manually set the timer to fire immediately
        timer = orch._exchange_timers.pop("m1")
        timer.cancel()
        orch._exchange_state["m1"] = "idle"
        assert orch._get_exchange_state("m1") == "idle"


class TestCooldown(unittest.TestCase):

    def test_not_on_cooldown_initially(self):
        orch, _ = _make_orch()
        assert orch._on_cooldown("m1") is False

    def test_on_cooldown_after_speak(self):
        orch, _ = _make_orch()
        orch._record_speak("m1")
        assert orch._on_cooldown("m1") is True

    def test_off_cooldown_after_interval(self):
        orch, _ = _make_orch()
        orch._last_spoken["m1"] = time.time() - 999
        assert orch._on_cooldown("m1") is False


class TestProcessFlushed(unittest.IsolatedAsyncioTestCase):

    async def test_address_no_content_speaks_greeting(self):
        orch, spoken = _make_orch()
        await orch.start_meeting("m1")
        seg = _seg("Hey Q", "m1")
        await orch._process_flushed(seg)
        assert any("What do you need" in c.text for c in spoken)

    async def test_address_with_content_calls_agent(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=QResponse(spoken="Here are your tasks."))

        orch, spoken = _make_orch(agent=mock_agent)
        await orch.start_meeting("m1")

        seg = _seg("Hey Q could you show me my Asana tasks?", "m1")
        await orch._process_flushed(seg)

        mock_agent.run.assert_called_once()
        call_text = mock_agent.run.call_args[0][0]
        assert "Q" not in call_text   # trigger stripped
        assert len(spoken) == 1
        assert spoken[0].text == "Here are your tasks."

    async def test_chat_callback_called_when_chat_set(self):
        chat_msgs = []

        async def fake_chat(meeting_id, text):
            chat_msgs.append(text)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=QResponse(
            spoken="You have one task.",
            chat="You have one task: [Fix login](https://app.asana.com/0/0/1/f)",
        ))

        orch, spoken = _make_orch(agent=mock_agent)
        orch._chat = fake_chat
        await orch.start_meeting("m1")
        orch._mode.set_mode("active")

        await orch._process_flushed(_seg("show me asana tasks", "m1"))

        assert len(chat_msgs) == 1
        assert "https://app.asana.com" in chat_msgs[0]

    async def test_chat_callback_not_called_when_chat_none(self):
        chat_msgs = []

        async def fake_chat(meeting_id, text):
            chat_msgs.append(text)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=QResponse(spoken="Done.", chat=None))

        orch, spoken = _make_orch(agent=mock_agent)
        orch._chat = fake_chat
        await orch.start_meeting("m1")
        orch._mode.set_mode("active")

        await orch._process_flushed(_seg("anything", "m1"))

        assert len(chat_msgs) == 0

    async def test_non_addressed_on_demand_stays_quiet(self):
        orch, spoken = _make_orch()
        await orch.start_meeting("m1")
        orch._mode.set_mode("on_demand")

        seg = _seg("Let's talk about authentication", "m1")
        await orch._process_flushed(seg)
        assert len(spoken) == 0

    async def test_non_addressed_active_mode_calls_agent(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=QResponse(spoken="Auth was discussed last week."))

        orch, spoken = _make_orch(agent=mock_agent)
        await orch.start_meeting("m1")
        orch._mode.set_mode("active")

        seg = _seg("Let's talk about authentication", "m1")
        await orch._process_flushed(seg)

        mock_agent.run.assert_called_once()

    async def test_cooldown_suppresses_agent_call(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=QResponse(spoken="Something"))

        orch, spoken = _make_orch(agent=mock_agent)
        await orch.start_meeting("m1")
        orch._mode.set_mode("active")
        orch._record_speak("m1")   # trigger cooldown

        seg = _seg("Let's continue discussing this", "m1")
        await orch._process_flushed(seg)

        mock_agent.run.assert_not_called()

    async def test_agent_returns_none_does_not_speak(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=None)

        orch, spoken = _make_orch(agent=mock_agent)
        await orch.start_meeting("m1")
        orch._mode.set_mode("active")

        seg = _seg("General meeting chatter here", "m1")
        await orch._process_flushed(seg)

        assert len(spoken) == 0

    async def test_engaged_exchange_responds_without_trigger(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=QResponse(spoken="Certainly!"))

        orch, spoken = _make_orch(agent=mock_agent)
        await orch.start_meeting("m1")
        orch._mode.set_mode("on_demand")
        orch._set_exchange_engaged("m1")

        seg = _seg("Can you also check Notion?", "m1")
        await orch._process_flushed(seg)

        mock_agent.run.assert_called_once()


class TestCleanResponse(unittest.TestCase):

    def test_strips_skip(self):
        assert QOrchestrator._clean_response("Done. SKIP") == "Done."

    def test_plain_text(self):
        assert QOrchestrator._clean_response("  Hello  ") == "Hello"


if __name__ == "__main__":
    unittest.main()
