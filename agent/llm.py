"""
agent/llm.py — Unified LLM factory for Quorum

Provides get_llm() for the primary LLM (Hermes via local Ollama) and
get_fallback_llm() for OpenRouter. The graph and context modules use
get_llm_with_fallback() which tries Hermes first, then OpenRouter.

Ollama exposes an OpenAI-compatible /v1 endpoint, so ChatOpenAI works
for both providers.
"""

import os
import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ── Hermes (local Ollama) ────────────────────────────────────────────────────
_HERMES_HOST  = os.getenv("HERMES_HOST", "http://localhost:11434")
_HERMES_MODEL = os.getenv("HERMES_MODEL", "hermes3")
_HERMES_TIMEOUT = float(os.getenv("HERMES_TIMEOUT", "3.0"))  # fast fail

# ── OpenRouter (cloud fallback) ──────────────────────────────────────────────
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini",  # fast, cheap, reliable tool-calling
)

QUORUM_SYSTEM_PROMPT = (
    "You are Q (short for Quorum), an AI meeting participant currently in a live meeting. "
    "People will address you as 'Q', 'Hey Q', or 'Quorum'. "
    "You are helpful, concise, and natural — like a knowledgeable teammate. "
    "Keep all spoken responses under 2 sentences.\n\n"
    "YOUR CAPABILITIES:\n"
    "- Answer questions using meeting context and conversation history\n"
    "- Search GitHub for PRs, issues, and code\n"
    "- Search Notion for docs and specs\n"
    "- Search Slack for messages and conversations\n"
    "- Search Asana for tasks and tickets\n"
    "- Search past meeting history for decisions and context\n"
    "- Create tasks and log decisions\n\n"
    "YOUR LIMITATIONS:\n"
    "- You CANNOT open or launch apps, browse the web, access personal "
    "calendars, send emails, or control anything on a user's computer.\n"
    "- You CAN search Slack, GitHub, Notion, and Asana for information — "
    "but you cannot open those apps or send messages in them.\n"
    "- If asked to do something you can't, say so clearly and suggest "
    "what you CAN do instead (e.g. 'I can't open Slack, but I can search "
    "Slack messages for you — what should I look for?').\n"
    "- Never make up capabilities you don't have.\n\n"
    "BEHAVIOUR:\n"
    "- If someone greets you, respond warmly and briefly.\n"
    "- If someone asks a question, use your tools to find the answer.\n"
    "- If the conversation is general chatter not directed at you, "
    "respond with exactly: SKIP\n"
    "- Reply only in English."
)


def get_hermes(temperature: float = 0.3) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance pointed at local Ollama (Hermes).

    Ollama serves an OpenAI-compatible API at /v1.
    """
    return ChatOpenAI(
        base_url=f"{_HERMES_HOST}/v1",
        model=_HERMES_MODEL,
        api_key="ollama",           # Ollama doesn't require a real key
        temperature=temperature,
        timeout=_HERMES_TIMEOUT,
        max_retries=0,              # fail fast — we'll fall back to OpenRouter
    )


def get_openrouter(temperature: float = 0.3, timeout: float = 15.0) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance pointed at OpenRouter (cloud fallback).
    """
    if not _OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY env var not set")

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=_OPENROUTER_MODEL,
        api_key=_OPENROUTER_API_KEY,
        temperature=temperature,
        timeout=timeout,
        max_retries=1,
    )


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    Return the primary LLM: Hermes (local Ollama).

    Use this when you want Hermes specifically (e.g. for tool-calling in the
    graph). Falls back handled at the call site or via get_llm_with_fallback().
    """
    return get_hermes(temperature=temperature)


async def ainvoke_with_fallback(prompt, *, temperature: float = 0.3) -> str:
    """
    Try Hermes first, fall back to OpenRouter. Returns plain text.

    Convenience for modules (like context.py) that just need a string response
    and don't need tool-calling.
    """
    try:
        hermes = get_hermes(temperature=temperature)
        response = await hermes.ainvoke(prompt)
        logger.debug("LLM: Hermes responded")
        return response.content.strip()
    except Exception as exc:
        logger.warning("Hermes unavailable (%s) — falling back to OpenRouter", exc)

    try:
        openrouter = get_openrouter(temperature=temperature)
        response = await openrouter.ainvoke(prompt)
        logger.debug("LLM: OpenRouter responded")
        return response.content.strip()
    except Exception as exc:
        logger.error("OpenRouter also failed: %s", exc)
        return "I wasn't able to process that right now."
