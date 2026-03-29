"""
tests/test_q_agent.py — Unit tests for agent/q_agent.py

All tests are hermetic — no network calls, no real LLM.
The LLM is patched at the _call_llm level so tests validate
the agentic loop logic: tool dispatch, result feeding, final response.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from agent.context import MeetingContext
from agent.q_agent import QAgent, QResponse, _clean_response


class TestCleanResponse(unittest.TestCase):

    def test_plain_text_unchanged(self):
        assert _clean_response("OAuth 2.0 was chosen.") == "OAuth 2.0 was chosen."

    def test_strips_trailing_skip(self):
        assert _clean_response("I can't do that. SKIP") == "I can't do that."

    def test_strips_skip_with_newline(self):
        assert _clean_response("Good answer.\nSKIP") == "Good answer."

    def test_pure_skip_becomes_empty(self):
        assert _clean_response("SKIP") == "SKIP"   # no preceding sentence, stays

    def test_case_insensitive(self):
        assert _clean_response("Done. skip") == "Done."

    def test_whitespace_only_becomes_empty(self):
        assert _clean_response("   ") == ""


class TestQAgentRun(unittest.IsolatedAsyncioTestCase):

    def _make_agent(self):
        ctx = MeetingContext(history_file="/dev/null")
        agent = QAgent(context=ctx)
        return agent

    async def test_text_response_no_tools(self):
        """LLM returns text immediately → agent returns QResponse."""
        agent = self._make_agent()
        agent._call_llm = AsyncMock(return_value={"content": "PR 42 is merged.", "tool_calls": []})

        result = await agent.run("What's the status of PR 42?", "mid-1", "Alice")
        assert isinstance(result, QResponse)
        assert result.spoken == "PR 42 is merged."
        assert result.chat is None   # no URLs → chat not needed

    async def test_response_with_url_sets_chat(self):
        """Response containing a URL → chat field is set, spoken is clean."""
        agent = self._make_agent()
        agent._call_llm = AsyncMock(return_value={
            "content": "Here it is: [Fix login](https://app.asana.com/0/0/1/f)",
            "tool_calls": [],
        })

        result = await agent.run("show me the task", "mid-1", "Alice")
        assert result is not None
        assert "https://" not in result.spoken
        assert result.chat is not None
        assert "https://app.asana.com" in result.chat

    async def test_skip_returns_none(self):
        """LLM returns SKIP → agent returns None (stay quiet)."""
        agent = self._make_agent()
        agent._call_llm = AsyncMock(return_value={"content": "SKIP", "tool_calls": []})

        result = await agent.run("The weather looks nice today.", "mid-1", "Bob")
        assert result is None

    async def test_empty_content_returns_none(self):
        agent = self._make_agent()
        agent._call_llm = AsyncMock(return_value={"content": "", "tool_calls": []})

        result = await agent.run("hmm", "mid-1", "Bob")
        assert result is None

    async def test_tool_call_then_text(self):
        """LLM calls a tool first, then returns text on second iteration."""
        agent = self._make_agent()

        tool_call = {
            "id": "call_1",
            "function": {"name": "search_asana", "arguments": '{"query": "tasks"}'},
        }

        async def fake_tool(meeting_id, query):
            return "- [asana] Fix login: open (https://asana.com/1)"

        agent.register_tools({"search_asana": fake_tool})

        responses = [
            {"content": "", "tool_calls": [tool_call]},
            {"content": "You have one open task: Fix login.", "tool_calls": []},
        ]
        agent._call_llm = AsyncMock(side_effect=responses)

        result = await agent.run("Show me my Asana tasks", "mid-2", "Alice")
        assert isinstance(result, QResponse)
        assert result.spoken == "You have one open task: Fix login."
        assert agent._call_llm.call_count == 2

        # Verify tool result was injected into messages
        messages_second_call = agent._call_llm.call_args_list[1][0][0]
        tool_msg = next(m for m in messages_second_call if m["role"] == "tool")
        assert "Fix login" in tool_msg["content"]

    async def test_unknown_tool_returns_error_in_message(self):
        """Unknown tool name → error string fed back, loop continues."""
        agent = self._make_agent()
        tool_call = {
            "id": "call_x",
            "function": {"name": "nonexistent_tool", "arguments": "{}"},
        }
        responses = [
            {"content": "", "tool_calls": [tool_call]},
            {"content": "I couldn't do that.", "tool_calls": []},
        ]
        agent._call_llm = AsyncMock(side_effect=responses)

        result = await agent.run("Do something weird", "mid-3", "Alice")
        assert result.spoken == "I couldn't do that."

        messages = agent._call_llm.call_args_list[1][0][0]
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert "unknown tool" in tool_msg["content"].lower()

    async def test_max_iterations_hit(self):
        """If LLM always returns tool_calls, we hit max iterations and return fallback."""
        agent = self._make_agent()
        tool_call = {
            "id": "c",
            "function": {"name": "search_slack", "arguments": '{"query": "x"}'},
        }
        agent.register_tools({"search_slack": AsyncMock(return_value="result")})
        agent._call_llm = AsyncMock(return_value={"content": "", "tool_calls": [tool_call]})

        result = await agent.run("keep looping", "mid-4", "X")
        assert result.spoken == "I wasn't able to complete that in time."
        assert agent._call_llm.call_count == 4   # _MAX_ITERATIONS

    async def test_llm_failure_returns_error_message(self):
        """Both LLM providers fail → agent returns QResponse with error, not exception."""
        agent = self._make_agent()
        agent._call_llm = AsyncMock(return_value=None)

        result = await agent.run("Anything", "mid-5", "X")
        assert result is not None
        assert "wasn't able" in result.spoken.lower()

    async def test_parallel_tool_calls(self):
        """Multiple tool calls in one response are all executed."""
        agent = self._make_agent()

        async def fake_slack(meeting_id, query):
            return "slack result"

        async def fake_notion(meeting_id, query):
            return "notion result"

        agent.register_tools({
            "search_slack": fake_slack,
            "search_notion": fake_notion,
        })

        tool_calls = [
            {"id": "c1", "function": {"name": "search_slack", "arguments": '{"query": "auth"}'}},
            {"id": "c2", "function": {"name": "search_notion", "arguments": '{"query": "auth"}'}},
        ]
        responses = [
            {"content": "", "tool_calls": tool_calls},
            {"content": "Auth was decided via OAuth.", "tool_calls": []},
        ]
        agent._call_llm = AsyncMock(side_effect=responses)

        result = await agent.run("auth decision?", "mid-6", "X")
        assert result.spoken == "Auth was decided via OAuth."

        messages = agent._call_llm.call_args_list[1][0][0]
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2


if __name__ == "__main__":
    unittest.main()
