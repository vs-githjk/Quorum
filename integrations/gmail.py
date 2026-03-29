"""
integrations/gmail.py — Gmail API client for Quorum.

Uses OAuth2 refresh token to access Gmail REST API.
Supports searching the inbox and sending emails.

Required env vars:
    GMAIL_CLIENT_ID      — OAuth2 client ID from Google Cloud Console
    GMAIL_CLIENT_SECRET  — OAuth2 client secret from Google Cloud Console
    GMAIL_REFRESH_TOKEN  — Refresh token from one-time OAuth flow (get_gmail_token.py)
    GMAIL_USER_EMAIL     — Gmail address (e.g. quorumtester412@gmail.com)
"""

import asyncio
import base64
import email as _email_lib
import logging
import os
import time
from email.mime.text import MIMEText

import aiohttp

from agent import IntegrationResult

logger = logging.getLogger(__name__)

_CLIENT_ID     = os.getenv("GMAIL_CLIENT_ID", "")
_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET", "")
_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN", "")
_USER_EMAIL    = os.getenv("GMAIL_USER_EMAIL", "")

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_API_BASE  = "https://gmail.googleapis.com/gmail/v1"

# In-memory token cache: {"access_token": str, "expires_at": float}
_token_cache: dict = {}


async def _get_access_token() -> str | None:
    """
    Return a valid access token, refreshing if expired.
    Caches the token in memory for its lifetime (~1 hour).
    """
    now = time.time()
    if _token_cache.get("access_token") and _token_cache.get("expires_at", 0) > now + 60:
        return _token_cache["access_token"]

    if not all([_CLIENT_ID, _CLIENT_SECRET, _REFRESH_TOKEN]):
        logger.warning("Gmail: GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET / GMAIL_REFRESH_TOKEN not set")
        return None

    payload = {
        "client_id":     _CLIENT_ID,
        "client_secret": _CLIENT_SECRET,
        "refresh_token": _REFRESH_TOKEN,
        "grant_type":    "refresh_token",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(_TOKEN_URL, data=payload) as resp:
                if not resp.ok:
                    text = await resp.text()
                    logger.warning("Gmail: token refresh failed HTTP %d: %s", resp.status, text[:200])
                    return None
                data = await resp.json()
    except Exception as exc:
        logger.warning("Gmail: token refresh error: %s", exc)
        return None

    token = data.get("access_token")
    expires_in = data.get("expires_in", 3600)
    if token:
        _token_cache["access_token"] = token
        _token_cache["expires_at"]   = now + expires_in
    return token


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
    }


def _snippet_to_result(msg: dict, full: dict | None = None) -> IntegrationResult:
    """Build an IntegrationResult from a Gmail message dict."""
    msg_id  = msg.get("id", "")
    snippet = msg.get("snippet", "")

    subject = sender = date_str = ""
    if full:
        headers = full.get("payload", {}).get("headers", [])
        for h in headers:
            n = h.get("name", "").lower()
            if n == "subject":
                subject = h.get("value", "")
            elif n == "from":
                sender = h.get("value", "")
            elif n == "date":
                date_str = h.get("value", "")

    title   = subject or f"Email {msg_id[:8]}"
    summary = f"From: {sender}. {date_str}. {snippet[:200]}" if sender else snippet[:200]
    url     = f"https://mail.google.com/mail/u/0/#inbox/{msg_id}"

    return IntegrationResult(
        source="gmail",
        title=title,
        url=url,
        summary=summary.strip(),
        raw_data=full or msg,
        timestamp=time.time(),
    )


async def search_gmail(query: str, max_results: int = 3) -> list[IntegrationResult]:
    """
    Search Gmail inbox for messages matching the query.

    Args:
        query:       Search term (same syntax as Gmail search bar).
        max_results: Maximum number of results to return.

    Returns:
        List of IntegrationResult objects. Empty list on any error.
    """
    if not _USER_EMAIL:
        logger.warning("Gmail: GMAIL_USER_EMAIL not set — skipping")
        return []

    token = await _get_access_token()
    if not token:
        return []

    # Step 1: list matching message IDs
    list_url = f"{_API_BASE}/users/{_USER_EMAIL}/messages"
    params   = {"q": query, "maxResults": str(max_results)}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(list_url, headers=_headers(token), params=params) as resp:
                if not resp.ok:
                    logger.warning("Gmail search HTTP %d", resp.status)
                    return []
                data = await resp.json()
    except Exception as exc:
        logger.warning("Gmail search failed: %s", exc)
        return []

    messages = data.get("messages", [])[:max_results]
    if not messages:
        logger.info("Gmail: no results for query %r", query)
        return []

    # Step 2: fetch full headers for each message in parallel
    async def _fetch_full(msg_id: str) -> dict | None:
        url = f"{_API_BASE}/users/{_USER_EMAIL}/messages/{msg_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=_headers(token),
                    params={"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]},
                ) as resp:
                    if not resp.ok:
                        return None
                    return await resp.json()
        except Exception:
            return None

    fulls = await asyncio.gather(*[_fetch_full(m["id"]) for m in messages])

    results = []
    for msg, full in zip(messages, fulls):
        results.append(_snippet_to_result(msg, full))

    logger.info("Gmail: returned %d results for %r", len(results), query)
    return results


async def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email from the configured Gmail account.

    Args:
        to:      Recipient email address.
        subject: Email subject line.
        body:    Plain-text email body.

    Returns:
        Human-readable confirmation string, or error string on failure.
    """
    if not _USER_EMAIL:
        return "Error: GMAIL_USER_EMAIL not set."

    token = await _get_access_token()
    if not token:
        return "Error: could not obtain Gmail access token. Check GMAIL_CLIENT_ID / SECRET / REFRESH_TOKEN."

    # Build RFC 2822 message
    mime_msg = MIMEText(body, "plain")
    mime_msg["to"]      = to
    mime_msg["from"]    = _USER_EMAIL
    mime_msg["subject"] = subject

    raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()

    send_url = f"{_API_BASE}/users/{_USER_EMAIL}/messages/send"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                send_url,
                headers={**_headers(token), "Content-Type": "application/json"},
                json={"raw": raw},
            ) as resp:
                if not resp.ok:
                    text = await resp.text()
                    logger.warning("Gmail send failed HTTP %d: %s", resp.status, text[:200])
                    return f"Error: failed to send email to {to}."
                data = await resp.json()
    except Exception as exc:
        logger.warning("Gmail send error: %s", exc)
        return f"Error sending email: {exc}"

    msg_id = data.get("id", "unknown")
    logger.info("Gmail: sent email to %r (id=%s)", to, msg_id)
    return f"Email sent to {to} with subject '{subject}'."
