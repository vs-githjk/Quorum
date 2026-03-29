"""
agent/q_agent.py — Agentic core for Q.

Replaces the keyword-router + fixed intent handlers with a true tool-calling
LLM loop. Q decides what tools to call, executes them, feeds results back,
and produces a final spoken response.

Loop:
    user text
        → LLM(system_prompt, conversation_history, tools)
        → tool_call?  →  execute tool  →  feed result back  →  LLM
        → final text response  →  speak

Max iterations: 4 (prevents runaway loops)

LLM providers (tried in order):
    1. Hermes 3 via local Ollama  (/api/chat with tools)
    2. OpenRouter                 (same OpenAI-compatible format)
"""

import asyncio
import json
import logging
import os
import time
from typing import NamedTuple

import aiohttp

from .context import MeetingContext

logger = logging.getLogger(__name__)

# ── LLM config ────────────────────────────────────────────────────────────────

_HERMES_HOST    = os.getenv("HERMES_HOST", "http://localhost:11434")
_HERMES_CHAT    = f"{_HERMES_HOST}/api/chat"
_HERMES_MODEL   = os.getenv("HERMES_MODEL", "hermes3")
_HERMES_TIMEOUT = 8.0   # generous — tool calls need time

_OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

_MAX_ITERATIONS = 4

_SYSTEM_PROMPT = """\
You are Q, an AI participant embedded in a live meeting.

IMPORTANT: You MUST use the provided tools to answer questions. \
Never answer from memory or make up data. \
If someone asks for tasks, search results, PRs, or decisions — call the relevant tool first.

Tools available:
- get_screen_link         → get the noVNC screen sharing link to send to participants
- search_asana           → find Asana tasks (returns task_gid, name, due date, assignee, url)
- create_asana_task      → create a new Asana task
- update_asana_task      → change due date, name, notes, or assignee on an EXISTING task (use task_gid from search_asana)
- send_chat_message      → type a message (or URL) into the meeting chat
- search_slack           → search Slack messages
- search_notion          → search Notion docs
- search_github          → search GitHub PRs/issues
- search_gmail           → search Gmail inbox for emails matching a query
- send_email             → send an email from the Q Gmail account
- log_decision           → log a meeting decision
- search_past_meetings   → search past meeting history
- open_on_screen         → open a URL in the shared meeting browser (visible to all via noVNC)
- act_on_screen          → perform any task in the browser using vision (search, interact, fill forms, navigate)
- render_visualization   → generate and display a data visualization from meeting context
- draft_email            → compose and open a Gmail draft with a given subject and body
- create_calendar_event  → create a Google Calendar event with title, date, time, and optional guests
- summarize_meeting      → summarize the current meeting transcript into decisions, action items, and key points
- ask_claude             → ask Claude a question or get help with writing, reasoning, or analysis

Rules:
- ALWAYS call a tool before answering any question about tasks, docs, PRs, Slack, email, or past meetings.
- To send an email: call send_email with to, subject, and body. Confirm with the speaker before sending if the recipient or content is ambiguous.
- To check email: call search_gmail with a relevant query.
- To update a task: first call search_asana to get the task_gid, then call update_asana_task.
- NEVER create a new task when asked to update an existing one.
- When asked to send a link or URL to chat: call send_chat_message with the URL.
- When asked to open, show, pull up, or open in a new tab a URL or website: call open_on_screen (ignore "new tab" — just open it).
- When asked to search, interact with, or do something on a website: call act_on_screen with a clear instruction.
- When asked to CREATE, WRITE, or BUILD something (a doc, a report, a summary, a page): call act_on_screen or render_visualization — do NOT call search tools.
- After any screen action completes, always tell the user it is visible in the noVNC window and include the noVNC link in your response.
- When asked to "show" something that was already rendered or opened on screen: do NOT call any tools — just tell the user it is already visible in the noVNC window.
- When numbers, revenue, metrics, or data are discussed and a chart/visualization is requested: call render_visualization.
- When asked to write, send, compose, or draft an email: call draft_email.
- When asked to schedule, book, or create a meeting/event/call: call create_calendar_event.
- When asked to summarize the meeting, recap what was discussed, or generate meeting notes: call summarize_meeting.
- When asked for help writing something, reasoning through a problem, or anything that needs Claude's intelligence: call ask_claude.
- Keep spoken responses under 2 sentences after receiving tool results.
- If a tool returns no results, say so briefly: "Nothing came up for that" or "Couldn't find anything on that."
- Do not narrate tool use ("Let me search..." — just call the tool silently).
- NEVER include URLs, markdown links, or bullet formatting in your spoken response. Plain text only.
- When listing tasks or results, say only the names separated by commas.
- Reply in English only.
- Always address the speaker directly in second person ("I found...", "Done!", "Your tasks are..."). NEVER narrate or describe what the speaker is doing.
- If the message is not a direct request to you (Q), respond with exactly: SKIP

Personality and tone:
- You are a chill, sharp teammate — not a corporate assistant. Speak like a real person in a meeting.
- Use casual phrases: "Sure thing", "Got it", "Here's what I found", "Looks like...", "Yeah so...", "Done", "There you go", "Pulled that up for you", "All set".
- NEVER say "is now open on the screen" or "is visible in the noVNC window" the same way twice. Vary it: "Check the screen", "That's up now", "Take a look", "Got it on screen".
- NEVER say "I have completed the request" or "The page is now visible". Just say "Done" or "There you go".
- When you can't do something, be honest and short: "Can't get to that right now" not "I wasn't able to process that request at this time."
- Keep it tight. One sentence when possible. Two max.
"""

# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_screen_link",
            "description": "Get the noVNC screen sharing link to send to meeting participants.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_slack",
            "description": "Search Slack messages for a topic or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notion",
            "description": "Search Notion documents, specs, wikis, and notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_github",
            "description": "Search GitHub pull requests and issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PR number, branch name, or keyword"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_asana",
            "description": "Search Asana for tasks matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Task name or keyword"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_asana_task",
            "description": "Create a new task in Asana.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "notes": {"type": "string", "description": "Optional task description or notes"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_asana_task",
            "description": "Update fields on an existing Asana task (due date, name, notes, assignee). Use task_gid from search_asana results. Never use this to create a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_gid": {"type": "string", "description": "GID of the task to update"},
                    "due_on":   {"type": "string", "description": "New due date YYYY-MM-DD, or empty string to clear"},
                    "name":     {"type": "string", "description": "New task title"},
                    "notes":    {"type": "string", "description": "New task notes/description"},
                    "assignee": {"type": "string", "description": "'me' to assign to yourself, or a user GID"},
                },
                "required": ["task_gid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gmail",
            "description": "Search the Q Gmail inbox for emails matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term, e.g. 'from:bob subject:budget'"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email from the Q Gmail account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body":    {"type": "string", "description": "Plain-text email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_chat_message",
            "description": "Send a message or URL into the meeting chat so participants can see and click it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Text or URL to send into the meeting chat"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_decision",
            "description": "Log a decision that was made in the meeting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "description": "The decision text to log"}
                },
                "required": ["decision"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_past_meetings",
            "description": "Search notes and summaries from past meetings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or keyword to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_on_screen",
            "description": "Open a URL in the shared meeting browser screen, visible to all participants via noVNC. Use when asked to open, show, display, or pull up a website or URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to open"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "act_on_screen",
            "description": "Perform any task in the shared browser using a vision loop — search, click, fill forms, navigate, create docs, write content, or interact with any website. Use this for CREATE/WRITE/BUILD tasks and for anything beyond just opening a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Plain English instruction of what to do in the browser, e.g. 'search for quarterly revenue trends on Google' or 'find the pricing page on stripe.com'",
                    }
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_visualization",
            "description": "Generate and display a data visualization on the shared screen. Use when the meeting discusses numbers, metrics, revenue, timelines, or comparisons and a chart would help.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What the visualization should show, e.g. 'bar chart of quarterly revenue growth'",
                    },
                    "data": {
                        "type": "string",
                        "description": "The data to visualize, extracted from the conversation, e.g. 'Q1: 100k, Q2: 150k, Q3: 200k'",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Compose a Gmail draft and open it in the browser. Use when asked to write, draft, send, or compose an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body":    {"type": "string", "description": "Full email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a Google Calendar event. Use when asked to schedule, book, or create a meeting, call, or event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":  {"type": "string", "description": "Event title"},
                    "date":   {"type": "string", "description": "Date in natural language, e.g. 'Friday', 'March 31', 'tomorrow'"},
                    "time":   {"type": "string", "description": "Time in natural language, e.g. '2pm', '14:00', '3:30pm'"},
                    "guests": {"type": "string", "description": "Comma-separated guest email addresses (optional)"},
                },
                "required": ["title", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_meeting",
            "description": "Summarize the current meeting into decisions made, action items, and key discussion points. Use when asked to recap, summarize, or generate meeting notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Optional — what to focus on, e.g. 'action items only' or 'technical decisions'. Leave empty for a full summary.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_claude",
            "description": "Ask Claude for help with writing, analysis, reasoning, or any question. Use for drafting docs, PRDs, emails, brainstorming, or anything that needs AI assistance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The full question or request for Claude",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
]


def _clean_response(text: str) -> str:
    """Strip trailing SKIP marker the LLM may append to an otherwise valid sentence."""
    import re
    cleaned = re.sub(r"[\s\n]+SKIP\s*$", "", text.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_for_speech(text: str) -> str:
    """
    Remove markdown and URLs so TTS reads naturally.

    - [label](url)  → label
    - bare https?://... URLs → removed
    - **bold** / *italic*   → plain text
    - bullet lines (-, *, 1.) → comma-joined
    - excess whitespace / newlines → single spaces
    """
    import re
    # Markdown links → label only
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Bold / italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Bullet list lines → collect items, join with comma
    lines = text.splitlines()
    items, rest = [], []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-*]\s+|^\d+\.\s+", stripped):
            items.append(re.sub(r"^[-*\d.]+\s*", "", stripped))
        else:
            rest.append(stripped)
    if items:
        text = ", ".join(items) + (" " + " ".join(rest) if any(rest) else "")
    else:
        text = " ".join(rest)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class QResponse(NamedTuple):
    """
    Returned by QAgent.run().

    spoken: short, URL-free text for TTS.
    chat:   full response preserving markdown links for meeting chat.
            None when nothing should be sent to chat.
    """
    spoken: str
    chat: str | None = None


# ── QAgent ────────────────────────────────────────────────────────────────────

class QAgent:
    """
    Agentic loop for Q.

    Given a user utterance and meeting context, calls the LLM with tools,
    executes any tool calls, feeds results back, and returns the final
    spoken response (or None if nothing should be said).

    Usage:
        agent = QAgent(context=meeting_context)
        response = await agent.run(
            user_text="Could you give me all my Asana tasks?",
            meeting_id="abc123",
            speaker="Abhinav",
        )
        if response:
            await speak(response)
    """

    def __init__(self, context: MeetingContext) -> None:
        self._context = context
        # Tool executors are injected at runtime to avoid circular imports
        self._tool_fns: dict = {}

    def register_tools(self, tool_fns: dict) -> None:
        """
        Register async callables for each tool name.

        Args:
            tool_fns: dict mapping tool name → async callable.
                      e.g. {"search_slack": search_slack_fn, ...}
        """
        self._tool_fns = tool_fns

    async def run(
        self,
        user_text: str,
        meeting_id: str,
        speaker: str = "Participant",
    ) -> QResponse | None:
        """
        Run the agentic loop for a single user utterance.

        Args:
            user_text:  The spoken text to respond to.
            meeting_id: Active meeting session ID.
            speaker:    Speaker name for context.

        Returns:
            QResponse(spoken, chat), or None if Q should stay quiet.
            spoken — clean text for TTS (no URLs or markdown).
            chat   — full response with URLs for meeting chat, or None if
                     the spoken version already contains everything.
        """
        recent = self._context.get_recent_transcript(meeting_id, n=6)

        system_with_context = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Current meeting ID: {meeting_id}\n"
            f"Recent transcript:\n{recent}"
        )

        # Inject the last 3 exchanges so Q remembers recent tool results
        # (e.g. task_gid from a task just created, URLs, prior search results)
        history = self._context.get_agent_history(meeting_id, n=3)
        current_user_msg = {"role": "user", "content": f"{speaker}: {user_text}"}

        messages = [
            {"role": "system", "content": system_with_context},
            *history,
            current_user_msg,
        ]

        # Track where the current exchange starts so we can save just this turn
        exchange_start = 1 + len(history)

        for iteration in range(_MAX_ITERATIONS):
            response = await self._call_llm(messages)
            if response is None:
                logger.error("[%s] LLM call failed on iteration %d", meeting_id, iteration)
                return QResponse(spoken="I wasn't able to process that right now.")

            tool_calls = response.get("tool_calls") or []
            content    = (response.get("content") or "").strip()

            if not tool_calls:
                # No more tool calls — this is the final response
                if not content or content.upper() == "SKIP":
                    return None
                # Append the final assistant message before saving
                messages.append({"role": "assistant", "content": content})
                self._context.add_agent_exchange(meeting_id, messages[exchange_start:])
                cleaned = _clean_response(content)
                spoken  = _clean_for_speech(cleaned)
                # Only send to chat if the full version differs (i.e. has URLs/links)
                chat = cleaned if cleaned != spoken else None
                return QResponse(spoken=spoken, chat=chat)

            # Execute all tool calls in parallel
            logger.info(
                "[%s] Agent iteration %d — calling tools: %s",
                meeting_id,
                iteration,
                [tc["function"]["name"] for tc in tool_calls],
            )

            tool_results = await asyncio.gather(
                *[self._execute_tool(tc, meeting_id) for tc in tool_calls],
                return_exceptions=True,
            )

            # Add assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })

            # Add tool result messages
            for tc, result in zip(tool_calls, tool_results):
                tool_id   = tc.get("id", tc["function"]["name"])
                tool_name = tc["function"]["name"]
                if isinstance(result, Exception):
                    result_text = f"Error: {result}"
                else:
                    result_text = str(result)
                logger.info("[%s] Tool %r → %r", meeting_id, tool_name, result_text[:120])
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "name":         tool_name,
                    "content":      result_text,
                })

        logger.warning("[%s] Agent hit max iterations (%d)", meeting_id, _MAX_ITERATIONS)
        self._context.add_agent_exchange(meeting_id, messages[exchange_start:])
        return QResponse(spoken="I wasn't able to complete that in time.")

    async def _execute_tool(self, tool_call: dict, meeting_id: str) -> str:
        """Execute a single tool call and return a string result."""
        fn_info = tool_call.get("function", {})
        name    = fn_info.get("name", "")
        try:
            args = json.loads(fn_info.get("arguments", "{}"))
        except json.JSONDecodeError:
            return "Error: could not parse tool arguments."

        fn = self._tool_fns.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'."

        try:
            result = await fn(meeting_id=meeting_id, **args)
            return result
        except Exception as exc:
            logger.error("Tool %r raised: %s", name, exc)
            return f"Error running {name}: {exc}"

    # ── LLM callers ───────────────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict]) -> dict | None:
        """Try Hermes first, fall back to OpenRouter. Returns the message dict."""
        try:
            return await self._call_hermes(messages)
        except Exception as exc:
            logger.warning("Agent: Hermes unavailable (%s) — falling back to OpenRouter", exc)

        try:
            return await self._call_openrouter(messages)
        except Exception as exc:
            logger.error("Agent: OpenRouter also failed: %s", exc)
            return None

    async def _call_hermes(self, messages: list[dict]) -> dict:
        payload = {
            "model":    _HERMES_MODEL,
            "messages": messages,
            "tools":    TOOLS,
            "stream":   False,
        }
        timeout = aiohttp.ClientTimeout(total=_HERMES_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_HERMES_CHAT, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["message"]

    async def _call_openrouter(self, messages: list[dict]) -> dict:
        if not _OPENROUTER_KEY:
            raise ValueError("OPENROUTER_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {_OPENROUTER_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       _OPENROUTER_MODEL,
            "messages":    messages,
            "tools":       TOOLS,
            "tool_choice": "auto",
        }
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_OPENROUTER_URL, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]
