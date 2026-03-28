"""
integrations/notion.py — Notion API client for Quorum.

Searches the Notion workspace for pages matching the query, then fetches
a short text snippet from each page's content blocks.

Required env vars:
    NOTION_TOKEN — Notion integration secret (starts with "secret_").
                   Create at https://www.notion.so/my-integrations
                   and share relevant pages/databases with the integration.
"""

import asyncio
import logging
import os
import time

from agent import IntegrationResult
from .base import safe_get, safe_post

logger = logging.getLogger(__name__)

_NOTION_TOKEN   = os.getenv("NOTION_TOKEN", "")
_NOTION_VERSION = "2022-06-28"
_API_BASE       = "https://api.notion.com/v1"


def _headers() -> dict:
    return {
        "Authorization":  f"Bearer {_NOTION_TOKEN}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type":   "application/json",
    }


def _extract_title(page: dict) -> str:
    """Pull the page title from Notion's nested property structure."""
    props = page.get("properties", {})
    # Try common title property names
    for key in ("Name", "title", "Title"):
        prop = props.get(key, {})
        title_parts = prop.get("title", [])
        if title_parts:
            return "".join(p.get("plain_text", "") for p in title_parts).strip()
    # Fall back to page id
    return f"Notion page {page.get('id', 'unknown')[:8]}"


async def _fetch_page_snippet(page_id: str) -> str:
    """
    Fetch the first text block of a Notion page and return up to 200 chars.
    Returns empty string on any error — callers handle missing snippets gracefully.
    """
    url  = f"{_API_BASE}/blocks/{page_id}/children"
    data = await safe_get(url, _headers(), params={"page_size": "5"})
    if not isinstance(data, dict):
        return ""

    blocks = data.get("results", [])
    for block in blocks:
        block_type = block.get("type", "")
        content    = block.get(block_type, {})
        rich_texts = content.get("rich_text", [])
        text = "".join(rt.get("plain_text", "") for rt in rich_texts).strip()
        if text:
            return text[:200]
    return ""


async def search_notion(query: str, max_results: int = 3) -> list[IntegrationResult]:
    """
    Search the Notion workspace for pages matching the query.

    Fetches text snippets from each result page in parallel, then
    constructs IntegrationResult objects.

    Args:
        query:       Search term from the meeting transcript.
        max_results: Maximum number of results to return.

    Returns:
        List of IntegrationResult objects. Empty list on any error.
    """
    if not _NOTION_TOKEN:
        logger.warning("Notion: NOTION_TOKEN not set — skipping")
        return []

    body = {
        "query": query,
        "filter": {"value": "page", "property": "object"},
        "page_size": max_results,
    }
    data = await safe_post(f"{_API_BASE}/search", _headers(), body)

    if not isinstance(data, dict):
        logger.warning("Notion: search returned unexpected type: %s", type(data))
        return []

    pages = data.get("results", [])[:max_results]
    if not pages:
        logger.info("Notion: no results for query %r", query)
        return []

    # Fetch snippets for all pages in parallel
    snippets = await asyncio.gather(
        *[_fetch_page_snippet(p["id"]) for p in pages],
        return_exceptions=True,
    )

    results = []
    for page, snippet in zip(pages, snippets):
        if isinstance(snippet, Exception):
            snippet = ""

        title   = _extract_title(page)
        page_id = page.get("id", "")
        url     = page.get("url", f"https://notion.so/{page_id.replace('-', '')}")
        summary = snippet if snippet else f"Notion page: {title}"

        results.append(IntegrationResult(
            source="notion",
            title=title,
            url=url,
            summary=summary,
            raw_data=page,
            timestamp=time.time(),
        ))

    logger.info("Notion: returned %d results for %r", len(results), query)
    return results
