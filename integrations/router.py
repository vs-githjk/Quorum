"""
integrations/router.py — LLM-based integration source router.

Before fanning out to any external API, the IntegrationRouter asks the LLM
which sources are actually relevant to the query and how to refine the search
term for each one. This means:

  "pull up PR #123"    → GitHub only, query "123"
  "what did the spec say about auth" → Notion only, query "auth spec"
  "who mentioned the deadline"       → Slack only, query "deadline"
  "is there a task for the login bug"→ Asana only, query "login bug"

Falls back silently to the original sources/query if the LLM fails or returns
unparseable JSON — the integration still works, just without refinement.

LLM calls use OpenRouter. Required env var: OPENROUTER_API_KEY.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)

# ── LLM config (mirrors agent/orchestrator.py — intentional duplication) ──────

_OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

_SYSTEM_PROMPT = (
    "You are a JSON-only routing assistant. "
    "Reply ONLY with valid JSON. No explanation, no markdown."
)

_ROUTING_PROMPT_TEMPLATE = """\
You are routing a meeting search query to external work tools.
Decide which tools to query and how to refine the search term for each.

Query: "{query}"
Available sources: {sources}

Tool descriptions:
- github: code repositories, pull requests (PRs), issues, branches, commits, bug numbers
- google: web search, general knowledge, external information, look up anything on the internet
- notion: documents, specs, wikis, meeting notes, decisions, design docs
- slack: messages, conversations, channel posts, announcements, DMs
- asana: tasks, tickets, action items, project tracking, assignments

Rules:
1. Only include sources from the available list.
2. Refine the query to what you would actually type into that tool's search box.
3. If the query clearly targets one tool, return only that tool.
4. If ambiguous, include all plausible tools.
5. Reply with ONLY this JSON shape — nothing else:

{{"sources": ["github"], "queries": {{"github": "PR #123 auth refactor"}}}}
"""


@dataclass
class RoutingPlan:
    """
    The router's output: which sources to query and with what refined query.

    Attributes:
        sources:         Subset of the requested sources the LLM selected.
        refined_queries: source → refined search string.
    """
    sources: list[str]
    refined_queries: dict[str, str] = field(default_factory=dict)


class IntegrationRouter:
    """
    Uses the LLM to decide which integration sources are relevant and how
    to refine the search query for each source.

    Usage:
        router = IntegrationRouter()
        plan = await router.route(req)
        # plan.sources = ["github"]
        # plan.refined_queries = {"github": "PR #123"}
    """

    async def route(self, req) -> RoutingPlan:
        """
        Produce a RoutingPlan for a ContextRequest.

        Args:
            req: ContextRequest with .query, .sources

        Returns:
            RoutingPlan. Falls back to original sources + query on LLM failure.
        """
        fallback = RoutingPlan(
            sources=list(req.sources),
            refined_queries={s: req.query for s in req.sources},
        )

        prompt = _ROUTING_PROMPT_TEMPLATE.format(
            query=req.query,
            sources=req.sources,
        )

        try:
            raw = await self._call_llm(prompt)
            plan = self._parse_response(raw, req)
            logger.info(
                "Router: sources=%s queries=%s",
                plan.sources,
                plan.refined_queries,
            )
            return plan
        except Exception as exc:
            logger.warning("Router: LLM routing failed (%s) — using fallback", exc)
            return fallback

    def _parse_response(self, raw: str, req) -> RoutingPlan:
        """
        Parse LLM JSON response into a RoutingPlan.

        Strips markdown fences if the model wraps its output.
        Falls back to original plan on parse error.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines
                if not l.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Router: JSON parse failed: %s — raw: %r", exc, raw[:200])
            raise

        raw_sources = data.get("sources", [])
        raw_queries = data.get("queries", {})

        # Validate: only keep sources that were in the original request
        valid_sources = [s for s in raw_sources if s in req.sources]
        if not valid_sources:
            logger.warning(
                "Router: LLM returned no valid sources %s — falling back to all", raw_sources
            )
            valid_sources = list(req.sources)

        refined = {}
        for s in valid_sources:
            refined[s] = raw_queries.get(s, req.query)

        return RoutingPlan(sources=valid_sources, refined_queries=refined)

    # ── LLM calls ─────────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        """Call OpenRouter."""
        return await self._call_openrouter(prompt)

    async def _call_openrouter(self, prompt: str) -> str:
        if not _OPENROUTER_KEY:
            raise ValueError("OPENROUTER_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {_OPENROUTER_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": _OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL, json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
