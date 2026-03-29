"""
integrations/asana.py — Asana API client for Quorum.

Search: uses the /tasks/search endpoint for full-text search across task
names and notes (not just prefix-matching like typeahead).

Write: create_asana_task() creates a real task in the workspace.

Required env vars:
    ASANA_TOKEN         — Personal access token from https://app.asana.com/0/my-apps
    ASANA_WORKSPACE_GID — Workspace GID (find at https://app.asana.com/api/1.0/workspaces)
"""

import logging
import os
import time

from agent import IntegrationResult
from .base import safe_get, safe_post

logger = logging.getLogger(__name__)

_ASANA_TOKEN         = os.getenv("ASANA_TOKEN", "")
_ASANA_WORKSPACE_GID = os.getenv("ASANA_WORKSPACE_GID", "")
_API_BASE            = "https://app.asana.com/api/1.0"

_TASK_FIELDS = "name,notes,permalink_url,assignee.name,due_on,completed"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_ASANA_TOKEN}",
        "Accept":        "application/json",
    }


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
    Search Asana for tasks matching the query using full-text search.

    Uses /workspaces/{gid}/tasks/search which searches task names AND notes,
    unlike typeahead which only prefix-matches names.

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

    url    = f"{_API_BASE}/workspaces/{_ASANA_WORKSPACE_GID}/tasks/search"
    params = {
        "text":       query,
        "opt_fields": _TASK_FIELDS,
        "limit":      str(max_results),
    }
    data = await safe_get(url, _headers(), params)

    if not isinstance(data, dict):
        logger.warning("Asana: search returned unexpected type: %s", type(data))
        return []

    tasks = data.get("data", [])[:max_results]
    if not tasks:
        logger.info("Asana: no results for query %r", query)
        return []

    results = [_task_to_result(t) for t in tasks]
    logger.info("Asana: returned %d results for %r", len(results), query)
    return results


async def create_asana_task(
    title: str,
    notes: str = "",
    assignee: str = "me",
) -> dict | None:
    """
    Create a new task in the configured Asana workspace.

    Args:
        title:    Task name.
        notes:    Optional task description / notes.
        assignee: Asana user GID or "me" (default: assign to token owner).

    Returns:
        The created task dict on success (includes permalink_url), None on failure.
    """
    if not _ASANA_TOKEN or not _ASANA_WORKSPACE_GID:
        logger.warning("Asana: cannot create task — token or workspace GID missing")
        return None

    body: dict = {
        "data": {
            "name":      title,
            "workspace": _ASANA_WORKSPACE_GID,
            "assignee":  assignee,
        }
    }
    if notes:
        body["data"]["notes"] = notes

    data = await safe_post(f"{_API_BASE}/tasks", _headers(), body)

    if not isinstance(data, dict) or "data" not in data:
        logger.warning("Asana: create task failed — response: %s", data)
        return None

    task = data["data"]
    logger.info("Asana: created task %r — %s", task.get("name"), task.get("permalink_url"))
    return task
