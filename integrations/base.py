"""
integrations/base.py — Shared aiohttp helpers for all integration clients.

All API calls in the integrations package go through safe_get / safe_post.
Both functions catch every exception, log it, and return None — callers
always check for None and return [] rather than propagating errors.
The 3-second default timeout prevents any single slow API from blocking
the meeting.
"""

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3.0  # seconds — keeps meetings responsive


async def safe_get(
    url: str,
    headers: dict,
    params: dict | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Any | None:
    """
    Perform an async HTTP GET and return the parsed JSON body.

    Args:
        url:     Full URL to GET.
        headers: Request headers (auth, content-type, etc.).
        params:  Query parameters dict (optional).
        timeout: Seconds before giving up (default 3.0).

    Returns:
        Parsed JSON (dict or list) on success, None on any error.
    """
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url, headers=headers, params=params or {}) as resp:
                if not resp.ok:
                    logger.warning(
                        "safe_get %s returned HTTP %d", url, resp.status
                    )
                    return None
                return await resp.json()
    except Exception as exc:
        logger.warning("safe_get %s failed: %s", url, exc)
        return None


async def safe_put(
    url: str,
    headers: dict,
    json_body: dict,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Any | None:
    """Async HTTP PUT with JSON body. Returns parsed JSON or None on error."""
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.put(url, headers=headers, json=json_body) as resp:
                if not resp.ok:
                    logger.warning("safe_put %s returned HTTP %d", url, resp.status)
                    return None
                return await resp.json()
    except Exception as exc:
        logger.warning("safe_put %s failed: %s", url, exc)
        return None


async def safe_post(
    url: str,
    headers: dict,
    json_body: dict,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Any | None:
    """
    Perform an async HTTP POST with a JSON body and return the parsed response.

    Args:
        url:       Full URL to POST.
        headers:   Request headers (auth, content-type, etc.).
        json_body: Body to send as JSON.
        timeout:   Seconds before giving up (default 3.0).

    Returns:
        Parsed JSON (dict or list) on success, None on any error.
    """
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                url, headers=headers, json=json_body
            ) as resp:
                if not resp.ok:
                    logger.warning(
                        "safe_post %s returned HTTP %d", url, resp.status
                    )
                    return None
                return await resp.json()
    except Exception as exc:
        logger.warning("safe_post %s failed: %s", url, exc)
        return None
