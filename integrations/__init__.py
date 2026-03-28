"""
integrations/__init__.py — Public entry point for the Quorum integrations layer.

Exposes a single async function:

    integration_callback(req: ContextRequest) -> list[IntegrationResult]

This is the function you pass to QuorumOrchestrator as integration_callback:

    from integrations import integration_callback

    orch = QuorumOrchestrator(
        speak_callback=speak_fn,
        integration_callback=integration_callback,
        action_callback=action_fn,
    )

Internally:
  1. IntegrationRouter uses the LLM to decide which sources to query and
     refines the search term per source.
  2. All selected sources are queried in parallel (asyncio.gather).
  3. Individual source failures are caught and logged — other sources
     still return results.
"""

import asyncio
import logging

from agent import ContextRequest, IntegrationResult

from .asana import search_asana
from .github import search_github
from .notion import search_notion
from .router import IntegrationRouter
from .slack import search_slack

logger = logging.getLogger(__name__)

# Source name → search function
_SOURCE_FN = {
    "github": search_github,
    "notion": search_notion,
    "slack":  search_slack,
    "asana":  search_asana,
}


async def integration_callback(req: ContextRequest) -> list[IntegrationResult]:
    """
    Fetch relevant context from external integrations for a ContextRequest.

    Called by QuorumOrchestrator when a topic is mentioned or a question
    needs live context.

    Args:
        req: ContextRequest with .query, .sources, .max_results, .meeting_id

    Returns:
        Flat list of IntegrationResult objects from all sources combined.
        Always returns a list (never None). Empty list if nothing found.
    """
    # ── Route: LLM decides which sources + refines the query ─────────────────
    router = IntegrationRouter()
    plan   = await router.route(req)

    logger.info(
        "[%s] Integration routing: sources=%s query=%r",
        req.meeting_id,
        plan.sources,
        req.query,
    )

    # ── Fan out to all selected sources in parallel ───────────────────────────
    tasks = []
    for source in plan.sources:
        fn = _SOURCE_FN.get(source)
        if fn is None:
            logger.warning("Unknown integration source: %r — skipping", source)
            continue
        refined_query = plan.refined_queries.get(source, req.query)
        tasks.append(fn(refined_query, req.max_results))

    if not tasks:
        return []

    batches = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Flatten results, log per-source errors ────────────────────────────────
    results: list[IntegrationResult] = []
    for source, batch in zip(plan.sources, batches):
        if isinstance(batch, Exception):
            logger.error(
                "[%s] Integration source %r raised: %s",
                req.meeting_id, source, batch,
            )
            continue
        if isinstance(batch, list):
            results.extend(batch)

    logger.info(
        "[%s] Integration returned %d total results",
        req.meeting_id,
        len(results),
    )
    return results
