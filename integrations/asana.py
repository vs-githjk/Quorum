"""
integrations/asana.py — Asana API client for Quorum.

Uses the typeahead endpoint to find tasks matching the query, then fetches
full task details in parallel.

Required env vars:
    ASANA_TOKEN         — Personal access token from https://app.asana.com/0/my-apps
    ASANA_WORKSPACE_GID — Workspace GID (find it at https://app.asana.com/api/1.0/workspaces)
"""

import asyncio
import logging
import os
import time

from agent import IntegrationResult
from .base import safe_get, safe_post, safe_put

logger = logging.getLogger(__name__)

_ASANA_TOKEN         = os.getenv("ASANA_TOKEN", "")
_ASANA_WORKSPACE_GID = os.getenv("ASANA_WORKSPACE_GID", "")
_ASANA_PROJECT_GID   = os.getenv("ASANA_PROJECT_GID", "")   # optional: puts tasks in a project
_API_BASE            = "https://app.asana.com/api/1.0"

_TASK_FIELDS = "gid,name,notes,permalink_url,assignee.name,due_on,completed,custom_fields"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_ASANA_TOKEN}",
        "Accept":        "application/json",
    }


async def _fetch_task_detail(task_gid: str) -> dict | None:
    """Fetch full task detail for a single task GID."""
    url  = f"{_API_BASE}/tasks/{task_gid}"
    data = await safe_get(url, _headers(), params={"opt_fields": _TASK_FIELDS})
    if isinstance(data, dict):
        return data.get("data")
    return None


def _task_to_result(task: dict) -> IntegrationResult:
    """Map an Asana task dict to an IntegrationResult."""
    name      = task.get("name", "Untitled task")
    notes     = (task.get("notes") or "")[:120]
    url       = task.get("permalink_url", "")
    completed = task.get("completed", False)
    due_on    = task.get("due_on") or "no due date"

    assignee      = task.get("assignee") or {}
    assignee_name = assignee.get("name", "unassigned")

    status  = "completed" if completed else "open"
    summary = f"Task: {name}. Assignee: {assignee_name}. Due: {due_on}. Status: {status}."
    if notes:
        summary += f" {notes}"

    return IntegrationResult(
        source="asana",
        title=name,
        url=url,
        summary=summary.strip(),
        raw_data=task,
        timestamp=time.time(),
    )


async def search_asana(query: str, max_results: int = 3) -> list[IntegrationResult]:
    """
    Search Asana for tasks matching the query using the typeahead endpoint.

    Fetches full task details in parallel after the typeahead returns stubs.

    Args:
        query:       Search term from the meeting transcript.
        max_results: Maximum number of results to return.

    Returns:
        List of IntegrationResult objects. Empty list on any error.
    """
    if not _ASANA_TOKEN:
        logger.warning("Asana: ASANA_TOKEN not set — skipping")
        return []

    if not _ASANA_WORKSPACE_GID:
        logger.warning("Asana: ASANA_WORKSPACE_GID not set — skipping")
        return []

    # The typeahead endpoint does name-prefix matching, not natural-language search.
    # Strip generic words so "all my tasks" → "" (returns all recent tasks).
    _GENERIC = frozenset({
        "all", "my", "our", "the", "a", "an", "tasks", "task",
        "asana", "show", "list", "get", "give", "me", "open", "current",
    })
    meaningful = " ".join(w for w in query.lower().split() if w not in _GENERIC)

    url    = f"{_API_BASE}/workspaces/{_ASANA_WORKSPACE_GID}/typeahead"
    params = {
        "resource_type": "task",
        "query":         meaningful,
        "count":         str(max_results),
        "opt_fields":    "gid,name",
    }
    data = await safe_get(url, _headers(), params)

    if not isinstance(data, dict):
        logger.warning("Asana: typeahead returned unexpected type: %s", type(data))
        return []

    stubs = data.get("data", [])[:max_results]
    if not stubs:
        logger.info("Asana: no results for query %r", query)
        return []

    # Fetch full details for all stubs in parallel
    details = await asyncio.gather(
        *[_fetch_task_detail(s["gid"]) for s in stubs],
        return_exceptions=True,
    )

    results = []
    for detail in details:
        if isinstance(detail, Exception) or detail is None:
            continue
        results.append(_task_to_result(detail))

    logger.info("Asana: returned %d results for %r", len(results), query)
    return results


async def create_asana_task(
    title: str,
    notes: str = "",
    workspace_gid: str | None = None,
) -> str:
    """
    Create a new task in Asana.

    Args:
        title:         Task name.
        notes:         Optional task description / notes.
        workspace_gid: Workspace GID override (defaults to ASANA_WORKSPACE_GID env var).

    Returns:
        "Created task: {title} ({url})" on success, or an error string on failure.
    """
    if not _ASANA_TOKEN:
        return "Error: ASANA_TOKEN not set."

    wid = workspace_gid or _ASANA_WORKSPACE_GID
    if not wid:
        return "Error: ASANA_WORKSPACE_GID not set."

    url  = f"{_API_BASE}/tasks"
    body = {"data": {"name": title, "workspace": wid}}
    if notes:
        body["data"]["notes"] = notes
    if _ASANA_PROJECT_GID:
        body["data"]["projects"] = [_ASANA_PROJECT_GID]

    data = await safe_post(url, _headers(), body)
    if not isinstance(data, dict):
        logger.warning("Asana create_task: unexpected response: %s", data)
        return f"Error: failed to create task '{title}'."

    task = data.get("data", {})
    permalink = task.get("permalink_url", "")
    logger.info("Asana: created task %r → %s", title, permalink)
    return f"Created task: {title} ({permalink})" if permalink else f"Created task: {title}"


async def update_asana_task(
    task_gid: str,
    due_on: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    assignee: str | None = None,
) -> str:
    """
    Update fields on an existing Asana task.

    Args:
        task_gid: GID of the task to update (from search_asana results).
        due_on:   New due date in YYYY-MM-DD format, or null to clear it.
        name:     New task title.
        notes:    New task description.
        assignee: "me" to assign to the token owner, or a user GID.

    Returns:
        Human-readable confirmation string or error string.
    """
    if not _ASANA_TOKEN:
        return "Error: ASANA_TOKEN not set."

    fields: dict = {}
    if due_on is not None:
        fields["due_on"] = due_on
    if name is not None:
        fields["name"] = name
    if notes is not None:
        fields["notes"] = notes
    if assignee is not None:
        fields["assignee"] = assignee

    if not fields:
        return "Error: no fields specified to update."

    url  = f"{_API_BASE}/tasks/{task_gid}"
    data = await safe_put(url, _headers(), {"data": fields})

    if not isinstance(data, dict):
        logger.warning("Asana update_task: unexpected response for gid=%s", task_gid)
        return f"Error: failed to update task {task_gid}."

    task      = data.get("data", {})
    task_name = task.get("name", task_gid)
    parts = []
    if due_on   is not None: parts.append(f"due date → {due_on or 'cleared'}")
    if name     is not None: parts.append(f"name → {name}")
    if notes    is not None: parts.append(f"notes updated")
    if assignee is not None: parts.append(f"assignee → {assignee}")
    logger.info("Asana: updated task %r (%s): %s", task_name, task_gid, ", ".join(parts))
    return f"Updated task '{task_name}': {', '.join(parts)}."
