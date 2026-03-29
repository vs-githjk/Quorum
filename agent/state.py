"""
agent/state.py — LangGraph state schema for Quorum

Defines the TypedDict that flows through every node in the graph.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class QuorumState(TypedDict):
    # ── Input (set before graph invocation) ──────────────────────────────────
    segment_text: str           # the flushed transcript text
    speaker: str                # who said it
    meeting_id: str             # active meeting session

    # ── Context (set before graph invocation) ────────────────────────────────
    mode: str                   # "active" or "on_demand"
    is_addressed: bool          # was Quorum explicitly addressed?
    exchange_state: str         # "idle" or "engaged"
    recent_transcript: str      # last N segments formatted

    # ── Conversation memory ──────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Output (set by graph nodes) ──────────────────────────────────────────
    response_text: str          # what Quorum should say (or "")
    should_speak: bool          # whether to actually speak
    decisions_logged: list[str] # decisions recorded during this turn
