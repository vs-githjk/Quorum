"""
bridge/mapper.py — Converts Python agent objects to QuorumCard dicts.

QuorumCard is the shape the companion React app expects. This module is
pure functions with no I/O — easy to test in isolation.
"""

import time
import uuid

# Processing steps shown in the animated pill for each source
_STEPS = {
    "github":         ["Connecting to repository...", "Searching pull requests...", "Fetching details..."],
    "notion":         ["Connecting to Notion...", "Searching documents...", "Loading content..."],
    "slack":          ["Connecting to Slack...", "Searching messages...", "Fetching thread..."],
    "asana":          ["Connecting to Asana...", "Finding tasks...", "Loading details..."],
    "create_task":    ["Processing request...", "Creating task...", "Confirming assignment..."],
    "pull_pr":        ["Connecting to repository...", "Fetching pull request...", "Loading details..."],
    "generate_chart": ["Pulling data...", "Calculating metrics...", "Rendering visualization..."],
    "decision":       ["Analyzing discussion...", "Extracting decision...", "Logging to record..."],
}

_ACTION_SOURCE = {
    "create_task":    "asana",
    "pull_pr":        "github",
    "generate_chart": "chart",
}

_ACTION_BADGE = {
    "create_task":    "Task created",
    "pull_pr":        "PR fetched",
    "generate_chart": "Chart generated",
}


def result_to_card(result, triggered_by: str, meeting_id: str) -> dict:
    """
    Convert an IntegrationResult to a QuorumCard dict.

    Args:
        result:       IntegrationResult from the integrations layer.
        triggered_by: The query or transcript text that caused this lookup.
        meeting_id:   The active meeting session ID.

    Returns:
        QuorumCard-shaped dict ready to JSON-serialize and send to the companion.
    """
    steps = _STEPS.get(result.source, ["Fetching data...", "Processing...", "Loading..."])
    return {
        "id":               str(uuid.uuid4()),
        "event_type":       "context_surfaced",
        "source":           result.source,
        "title":            result.title,
        "summary":          result.summary,
        "url":              result.url,
        "triggered_by":     triggered_by,
        "timestamp":        int(result.timestamp * 1000),
        "meeting_id":       meeting_id,
        "processing_steps": steps,
    }


def action_to_card(req, meeting_id: str) -> dict:
    """
    Convert a dispatched ActionRequest to a QuorumCard dict.

    Args:
        req:        ActionRequest that was dispatched to the action_callback.
        meeting_id: The active meeting session ID.

    Returns:
        QuorumCard-shaped dict ready to JSON-serialize and send to the companion.
    """
    steps  = _STEPS.get(req.action_type, ["Processing...", "Executing...", "Done."])
    source = _ACTION_SOURCE.get(req.action_type, "asana")
    badge  = _ACTION_BADGE.get(req.action_type, "Action taken")

    params = req.parameters or {}
    if req.action_type == "create_task":
        title = params.get("title", "New task")
    elif req.action_type == "pull_pr":
        ref   = params.get("reference", "PR")
        title = f"PR: {ref}"
    elif req.action_type == "generate_chart":
        topic = params.get("topic", "Chart")
        title = f"Chart: {topic}"
    else:
        title = req.action_type.replace("_", " ").title()

    return {
        "id":               str(uuid.uuid4()),
        "event_type":       "action_taken",
        "source":           source,
        "title":            title,
        "summary":          req.context,
        "url":              "",
        "triggered_by":     req.context,
        "timestamp":        int(time.time() * 1000),
        "meeting_id":       meeting_id,
        "badge":            badge,
        "processing_steps": steps,
    }


def decision_to_card(meeting_id: str, triggered_by: str = "Meeting discussion") -> dict:
    """
    Emit a decision card when Quorum logs a decision.

    Args:
        meeting_id:   The active meeting session ID.
        triggered_by: The transcript text that triggered the decision.

    Returns:
        QuorumCard-shaped dict.
    """
    return {
        "id":               str(uuid.uuid4()),
        "event_type":       "decision_logged",
        "source":           "meeting",
        "title":            "Decision logged",
        "summary":          "A decision was detected and logged to the meeting record.",
        "url":              "",
        "triggered_by":     triggered_by,
        "timestamp":        int(time.time() * 1000),
        "meeting_id":       meeting_id,
        "processing_steps": _STEPS["decision"],
    }
