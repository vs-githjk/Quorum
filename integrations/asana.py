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
from .base import safe_get

logger = logging.getLogger(__name__)

_ASANA_TOKEN         = os.getenv("ASANA_TOKEN", "")
_ASANA_WORKSPACE_GID = os.getenv("ASANA_WORKSPACE_GID", "")
_API_BASE            = "https://app.asana.com/api/1.0"

_TASK_FIELDS = "name,notes,permalink_url,assignee.name,due_on,completed,custom_fields"


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

    url    = f"{_API_BASE}/workspaces/{_ASANA_WORKSPACE_GID}/typeahead"
    params = {
        "resource_type": "task",
        "query":         query,
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
