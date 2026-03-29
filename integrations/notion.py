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


def _extract_title(obj: dict) -> str:
    """Pull the title from a Notion page or database object."""
    obj_type = obj.get("object", "page")

    # Databases store their title at the top level as a rich_text array
    if obj_type == "database":
        parts = obj.get("title", [])
        text = "".join(p.get("plain_text", "") for p in parts).strip()
        if text:
            return text

    # Pages store their title inside properties
    props = obj.get("properties", {})
    for key in ("Name", "title", "Title", "Page"):
        prop = props.get(key, {})
        title_parts = prop.get("title", [])
        if title_parts:
            return "".join(p.get("plain_text", "") for p in title_parts).strip()

    return f"Notion {obj_type} {obj.get('id', 'unknown')[:8]}"


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
        "query":     query,
        "page_size": max_results,
    }
    data = await safe_post(f"{_API_BASE}/search", _headers(), body)

    if not isinstance(data, dict):
        logger.warning("Notion: search returned unexpected type: %s", type(data))
        return []

    objects = data.get("results", [])[:max_results]
    if not objects:
        logger.info("Notion: no results for query %r", query)
        return []

    # Only fetch snippets for pages (databases don't have block children)
    page_ids = [o["id"] for o in objects if o.get("object") == "page"]
    snippets_map: dict[str, str] = {}
    if page_ids:
        fetched = await asyncio.gather(
            *[_fetch_page_snippet(pid) for pid in page_ids],
            return_exceptions=True,
        )
        for pid, snippet in zip(page_ids, fetched):
            snippets_map[pid] = snippet if isinstance(snippet, str) else ""

    results = []
    for obj in objects:
        obj_id  = obj.get("id", "")
        title   = _extract_title(obj)
        url     = obj.get("url", f"https://notion.so/{obj_id.replace('-', '')}")
        snippet = snippets_map.get(obj_id, "")
        summary = snippet if snippet else f"Notion {obj.get('object', 'page')}: {title}"

        results.append(IntegrationResult(
            source="notion",
            title=title,
            url=url,
            summary=summary,
            raw_data=obj,
            timestamp=time.time(),
        ))

    logger.info("Notion: returned %d results for %r", len(results), query)
    return results
