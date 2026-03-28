"""
integrations/github.py — GitHub API client for Quorum.

Searches for PRs by number (direct lookup) or by keyword (search API).
Returns IntegrationResult objects ready for the orchestrator to consume.

Required env vars:
    GITHUB_TOKEN  — Personal access token or fine-grained PAT.
                    Works without it for public repos at lower rate limits.
    GITHUB_OWNER  — Org or user name (e.g. "acme-corp"). Used for direct PR lookup.
    GITHUB_REPO   — Repo name (e.g. "backend"). Used for direct PR lookup.
"""

import logging
import os
import re
import time

from agent import IntegrationResult
from .base import safe_get

logger = logging.getLogger(__name__)

_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
_GITHUB_OWNER = os.getenv("GITHUB_OWNER", "")
_GITHUB_REPO  = os.getenv("GITHUB_REPO", "")
_API_BASE      = "https://api.github.com"

# Matches "PR #123", "PR123", "#123", "pull request 123"
_PR_NUMBER_RE = re.compile(r"(?:pr\s*#?|#|pull\s+request\s+)(\d+)", re.IGNORECASE)


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if _GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {_GITHUB_TOKEN}"
    return h


def _pr_to_result(pr: dict) -> IntegrationResult:
    """Map a GitHub PR API object to an IntegrationResult."""
    number = pr.get("number", "?")
    title  = pr.get("title", "Untitled PR")
    state  = pr.get("state", "unknown")
    body   = (pr.get("body") or "")[:120]
    url    = pr.get("html_url", "")
    merged = pr.get("merged_at")

    status_str = "merged" if merged else state
    summary = f"PR #{number}: {title}. Status: {status_str}."
    if body:
        summary += f" {body}"

    return IntegrationResult(
        source="github",
        title=f"PR #{number}: {title}",
        url=url,
        summary=summary.strip(),
        raw_data=pr,
        timestamp=time.time(),
    )


async def search_github(query: str, max_results: int = 3) -> list[IntegrationResult]:
    """
    Search GitHub for PRs matching the query.

    Strategy:
    1. If the query contains a PR number, do a direct lookup on the configured
       repo (fast, exact match).
    2. Otherwise, use the GitHub search API to find matching issues/PRs.

    Args:
        query:       Search term from the meeting transcript.
        max_results: Maximum number of results to return.

    Returns:
        List of IntegrationResult objects. Empty list on any error.
    """
    # ── Direct PR number lookup ───────────────────────────────────────────────
    match = _PR_NUMBER_RE.search(query)
    if match and _GITHUB_OWNER and _GITHUB_REPO:
        pr_number = match.group(1)
        url  = f"{_API_BASE}/repos/{_GITHUB_OWNER}/{_GITHUB_REPO}/pulls/{pr_number}"
        data = await safe_get(url, _headers())
        if isinstance(data, dict) and "number" in data:
            logger.info("GitHub: direct PR #%s lookup succeeded", pr_number)
            return [_pr_to_result(data)]
        # Fall through to search if direct lookup failed (e.g. wrong repo)
        logger.warning("GitHub: direct PR #%s lookup returned nothing", pr_number)

    # ── Keyword search ────────────────────────────────────────────────────────
    search_query = query
    if _GITHUB_OWNER and _GITHUB_REPO:
        search_query += f" repo:{_GITHUB_OWNER}/{_GITHUB_REPO}"
    search_query += " is:pr"

    params = {
        "q":        search_query,
        "per_page": str(max_results),
        "sort":     "updated",
        "order":    "desc",
    }
    data = await safe_get(f"{_API_BASE}/search/issues", _headers(), params)

    if not isinstance(data, dict):
        logger.warning("GitHub: search returned unexpected type: %s", type(data))
        return []

    items = data.get("items", [])[:max_results]
    if not items:
        logger.info("GitHub: no results for query %r", query)
        return []

    results = []
    for item in items:
        # Search API returns issue-shaped objects; massage into PR shape
        pr_like = {
            "number":   item.get("number"),
            "title":    item.get("title"),
            "state":    item.get("state"),
            "body":     item.get("body"),
            "html_url": item.get("html_url"),
            "merged_at": None,  # not returned by search API
        }
        results.append(_pr_to_result(pr_like))

    logger.info("GitHub: returned %d results for %r", len(results), query)
    return results
