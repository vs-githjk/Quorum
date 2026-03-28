"""
integrations/slack.py — Slack API client for Quorum.

Searches Slack messages matching the query using the search.messages endpoint.

Required env vars:
    SLACK_BOT_TOKEN — xoxb- bot token.
                      Requires the `search:read` OAuth scope.
                      Note: search.messages requires a user token (xoxp-) in some
                      Slack plans. If results are empty, switch to a user token.
"""

import logging
import os
import time

from agent import IntegrationResult
from .base import safe_get

logger = logging.getLogger(__name__)

_SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
_API_BASE    = "https://slack.com/api"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_SLACK_TOKEN}",
        "Content-Type":  "application/json",
    }


async def search_slack(query: str, max_results: int = 3) -> list[IntegrationResult]:
    """
    Search Slack messages for the given query.

    Args:
        query:       Search term from the meeting transcript.
        max_results: Maximum number of results to return.

    Returns:
        List of IntegrationResult objects. Empty list on any error.
    """
    if not _SLACK_TOKEN:
        logger.warning("Slack: SLACK_BOT_TOKEN not set — skipping")
        return []

    params = {
        "query":    query,
        "count":    str(max_results),
        "sort":     "timestamp",
        "sort_dir": "desc",
    }
    data = await safe_get(f"{_API_BASE}/search.messages", _headers(), params)

    if not isinstance(data, dict):
        logger.warning("Slack: search returned unexpected type: %s", type(data))
        return []

    if not data.get("ok"):
        logger.warning("Slack: API error — %s", data.get("error", "unknown"))
        return []

    messages = (
        data.get("messages", {})
            .get("matches", [])[:max_results]
    )
    if not messages:
        logger.info("Slack: no results for query %r", query)
        return []

    results = []
    for msg in messages:
        channel  = msg.get("channel", {})
        ch_name  = channel.get("name", "unknown-channel")
        username = msg.get("username") or msg.get("user", "unknown")
        text     = (msg.get("text") or "")[:200]
        url      = msg.get("permalink", "")

        title   = f"#{ch_name} — {username}"
        summary = text if text else f"Message in #{ch_name}"

        # Parse Slack's ts (e.g. "1691234567.123456") to a float timestamp
        try:
            ts = float(msg.get("ts", time.time()))
        except (ValueError, TypeError):
            ts = time.time()

        results.append(IntegrationResult(
            source="slack",
            title=title,
            url=url,
            summary=summary,
            raw_data=msg,
            timestamp=ts,
        ))

    logger.info("Slack: returned %d results for %r", len(results), query)
    return results
