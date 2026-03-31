"""
integrations/google_search.py — Google Custom Search API client for Quorum.

Searches the web via Google's Programmable Search Engine.

Required env vars:
    GOOGLE_API_KEY       — API key from console.cloud.google.com
    GOOGLE_SEARCH_CX     — Search Engine ID from programmablesearchengine.google.com
"""

import logging
import os
import time

from agent import IntegrationResult
from .base import safe_get

logger = logging.getLogger(__name__)

_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX", "")
_API_URL   = "https://www.googleapis.com/customsearch/v1"


def _params(query: str, max_results: int) -> dict:
    return {
        "key": _API_KEY,
        "cx":  _SEARCH_CX,
        "q":   query,
        "num": str(min(max_results, 10)),  # API max is 10 per request
    }


async def search_google(query: str, max_results: int = 3) -> list[IntegrationResult]:
    """
    Search the web via Google Custom Search.

    Args:
        query:       Search term.
        max_results: Maximum number of results to return (max 10).

    Returns:
        List of IntegrationResult objects. Empty list on any error.
    """
    if not _API_KEY or not _SEARCH_CX:
        logger.warning("Google Search: GOOGLE_API_KEY or GOOGLE_SEARCH_CX not set — skipping")
        return []

    data = await safe_get(_API_URL, {}, _params(query, max_results))

    if not isinstance(data, dict):
        logger.warning("Google Search: unexpected response type: %s", type(data))
        return []

    items = data.get("items", [])[:max_results]
    if not items:
        logger.info("Google Search: no results for query %r", query)
        return []

    results = []
    for item in items:
        results.append(IntegrationResult(
            source="google",
            title=item.get("title", ""),
            url=item.get("link", ""),
            summary=item.get("snippet", ""),
            raw_data=item,
            timestamp=time.time(),
        ))

    logger.info("Google Search: returned %d results for %r", len(results), query)
    return results
