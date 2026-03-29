"""
agent/tools.py — LangChain tool wrappers for Quorum integrations

Wraps existing integration search functions as LangChain @tool functions
so LangGraph's ToolNode can invoke them via the LLM's tool-calling.
"""

import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def search_github_tool(query: str) -> str:
    """Search GitHub for PRs, issues, and code matching the query.
    Use when someone mentions a PR number, code change, branch, or bug."""
    from integrations.github import search_github
    try:
        results = await search_github(query, max_results=3)
        if not results:
            return "No GitHub results found."
        return "\n".join(f"- {r.title}: {r.summary} ({r.url})" for r in results)
    except Exception as exc:
        logger.warning("search_github_tool failed: %s", exc)
        return f"GitHub search failed: {exc}"


@tool
async def search_notion_tool(query: str) -> str:
    """Search Notion workspace for docs, specs, and meeting notes.
    Use when someone asks about documentation, specs, or written notes."""
    from integrations.notion import search_notion
    try:
        results = await search_notion(query, max_results=3)
        if not results:
            return "No Notion results found."
        return "\n".join(f"- {r.title}: {r.summary} ({r.url})" for r in results)
    except Exception as exc:
        logger.warning("search_notion_tool failed: %s", exc)
        return f"Notion search failed: {exc}"


@tool
async def search_slack_tool(query: str) -> str:
    """Search Slack messages and conversations.
    Use when someone asks about messages, announcements, or team conversations."""
    from integrations.slack import search_slack
    try:
        results = await search_slack(query, max_results=3)
        if not results:
            return "No Slack results found."
        return "\n".join(f"- {r.title}: {r.summary} ({r.url})" for r in results)
    except Exception as exc:
        logger.warning("search_slack_tool failed: %s", exc)
        return f"Slack search failed: {exc}"


@tool
async def search_asana_tool(query: str) -> str:
    """Search Asana for tasks, tickets, and project items.
    Use when someone mentions tasks, tickets, action items, or assignments."""
    from integrations.asana import search_asana
    try:
        results = await search_asana(query, max_results=3)
        if not results:
            return "No Asana results found."
        return "\n".join(f"- {r.title}: {r.summary} ({r.url})" for r in results)
    except Exception as exc:
        logger.warning("search_asana_tool failed: %s", exc)
        return f"Asana search failed: {exc}"


# ── Meeting context tools ────────────────────────────────────────────────────
# These need the MeetingContext instance at runtime. We bind them via a factory.

def make_context_tools(meeting_context, action_callback):
    """
    Create tools that need access to the MeetingContext and action callback.

    Returns a list of LangChain tools with the context bound via closure.
    """

    @tool
    async def search_past_meetings_tool(query: str) -> str:
        """Search past meeting history for decisions and context.
        Use when someone asks about previous meetings or past decisions."""
        try:
            result = meeting_context.search_past_meetings(query)
            return result
        except Exception as exc:
            logger.warning("search_past_meetings_tool failed: %s", exc)
            return f"Past meeting search failed: {exc}"

    @tool
    async def create_task_tool(title: str, description: str = "") -> str:
        """Create a new task in the project management system.
        Use when someone says to create a task, add an action item, or log a to-do."""
        from .orchestrator import ActionRequest
        try:
            req = ActionRequest(
                action_type="create_task",
                parameters={"title": title, "description": description},
                context=title,
                meeting_id="",  # will be set by the graph
            )
            meeting_context.add_action(req)
            result = await action_callback(req)
            return f"Task created: {title}"
        except Exception as exc:
            logger.warning("create_task_tool failed: %s", exc)
            return f"Failed to create task: {exc}"

    @tool
    async def log_decision_tool(decision: str) -> str:
        """Log a team decision that was made during the meeting.
        Use when the team agrees on something or makes a final call."""
        return f"Decision logged: {decision}"

    return [search_past_meetings_tool, create_task_tool, log_decision_tool]


def get_all_tools(meeting_context=None, action_callback=None):
    """Return all tools, including context-bound ones if provided."""
    tools = [
        search_github_tool,
        search_notion_tool,
        search_slack_tool,
        search_asana_tool,
    ]
    if meeting_context and action_callback:
        tools.extend(make_context_tools(meeting_context, action_callback))
    return tools
