"""
agent/graph.py — LangGraph agent for Quorum meeting bot

Implements a ReAct loop: agent → tools → agent → ... → respond → END
The LLM decides which tools to call (implicit intent classification)
and formulates natural-language responses for all cases.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import QuorumState
from .llm import get_hermes, get_openrouter, QUORUM_SYSTEM_PROMPT
from .tools import get_all_tools

logger = logging.getLogger(__name__)


def build_graph(meeting_context=None, action_callback=None):
    """
    Build and compile the Quorum LangGraph.

    Tries Hermes (local Ollama) first for the agent node. If Hermes fails,
    falls back to OpenRouter automatically.

    Args:
        meeting_context: MeetingContext instance (for past meeting search tool).
        action_callback: async callback for dispatching actions.

    Returns:
        Compiled StateGraph ready for .ainvoke().
    """
    tools = get_all_tools(meeting_context, action_callback)

    # Build both LLMs with tools bound — agent_node tries Hermes first
    hermes = get_hermes()
    hermes_with_tools = hermes.bind_tools(tools)

    openrouter = get_openrouter()
    openrouter_with_tools = openrouter.bind_tools(tools)

    # ── Node: agent (LLM decides what to do) ─────────────────────────────────

    async def agent_node(state: QuorumState) -> dict:
        system = SystemMessage(content=QUORUM_SYSTEM_PROMPT)

        context_block = (
            f"[Meeting context]\n"
            f"Mode: {state['mode']}, Exchange: {state['exchange_state']}\n"
            f"Speaker: {state['speaker']}\n"
            f"Recent transcript:\n{state['recent_transcript']}\n"
        )

        human = HumanMessage(content=(
            f"{context_block}\n"
            f"[New from {state['speaker']}]: \"{state['segment_text']}\"\n\n"
            f"Decide how to respond. Use tools if you need to look something up "
            f"or take an action. Then respond in 1-2 sentences.\n"
            f"If this is general conversation not directed at you, "
            f"respond with exactly: SKIP"
        ))

        # Build message list: system + prior conversation + new input
        all_messages = [system] + list(state.get("messages", [])) + [human]

        # Try Hermes first, fall back to OpenRouter
        try:
            response = await hermes_with_tools.ainvoke(all_messages)
            logger.debug("Agent node: Hermes responded")
        except Exception as exc:
            logger.warning("Hermes unavailable (%s) — falling back to OpenRouter", exc)
            response = await openrouter_with_tools.ainvoke(all_messages)
            logger.debug("Agent node: OpenRouter responded")

        return {"messages": [human, response]}

    # ── Node: tools (execute tool calls) ─────────────────────────────────────

    tool_node = ToolNode(tools)

    # ── Node: respond (extract final answer) ─────────────────────────────────

    async def respond_node(state: QuorumState) -> dict:
        last_msg = state["messages"][-1]
        text = getattr(last_msg, "content", "") or ""
        text = text.strip()

        should_speak = bool(text) and text.upper() != "SKIP"

        # Extract any decisions logged by the log_decision_tool
        decisions = []
        for msg in state.get("messages", []):
            content = getattr(msg, "content", "") or ""
            if content.startswith("Decision logged: "):
                decisions.append(content.replace("Decision logged: ", ""))

        return {
            "response_text": text if should_speak else "",
            "should_speak": should_speak,
            "decisions_logged": decisions,
        }

    # ── Conditional edge: tools or respond? ──────────────────────────────────

    def should_use_tools(state: QuorumState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "respond"

    # ── Build the graph ──────────────────────────────────────────────────────

    graph = StateGraph(QuorumState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_use_tools, {
        "tools": "tools",
        "respond": "respond",
    })
    graph.add_edge("tools", "agent")   # loop back after tool execution
    graph.add_edge("respond", END)

    compiled = graph.compile()
    logger.info("LangGraph compiled — %d tools bound", len(tools))
    return compiled
