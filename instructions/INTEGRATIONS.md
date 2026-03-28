# Quorum — Integrations Branch: Integration Guide

This document covers everything built on top of the `agent` branch in the `integrations` branch, and exactly what teammates need to know to integrate with or build on it.

The agent branch README is still the authoritative reference for the orchestrator, intent detection, and interface contracts. Read that first if you haven't.

---

## Table of Contents

1. [What Was Built](#what-was-built)
2. [Setup](#setup)
3. [How It Works](#how-it-works)
4. [Wiring Into the Full System](#wiring-into-the-full-system)
5. [New Orchestrator Behaviour](#new-orchestrator-behaviour)
6. [Environment Variables](#environment-variables)
7. [Running Tests](#running-tests)
8. [Gotchas & Watch Outs](#gotchas--watch-outs)

---

## What Was Built

### New package: `integrations/`

| File | Role |
|---|---|
| `integrations/__init__.py` | Entry point — exports `integration_callback` |
| `integrations/router.py` | LLM-based smart router — decides which sources to query and refines the search query per source |
| `integrations/base.py` | Shared aiohttp helpers (`safe_get`, `safe_post`) with 3s timeout |
| `integrations/github.py` | GitHub API client — PR search by number or keyword |
| `integrations/notion.py` | Notion API client — workspace page search |
| `integrations/slack.py` | Slack API client — message search |
| `integrations/asana.py` | Asana API client — task typeahead + detail fetch |

### Modified: `agent/`

| File | What changed |
|---|---|
| `agent/context.py` | Added `is_already_surfaced(url, meeting_id)` — deduplication check |
| `agent/orchestrator.py` | Added clarifying questions, proactive offer mode, cooldown, deduplication filter, `_speak_and_record` |

---

## Setup

```bash
git checkout integrations

pip install -r requirements.txt

cp .env.example .env
# Fill in all API keys — see Environment Variables section below
```

**Python 3.11+ required.**

---

## How It Works

### Smart Router

When the orchestrator calls `integration_callback`, the router runs first:

```
ContextRequest (query, sources)
  ↓
IntegrationRouter.route()
  → asks LLM: which sources are relevant? how to refine the query per source?
  → returns RoutingPlan: {sources, refined_queries}
  ↓
Fan out in parallel to selected sources
  → asyncio.gather(search_github, search_notion, ...)
  ↓
Flat list of IntegrationResult objects
```

**If the LLM call fails** (rate limit, no key, timeout) — the router falls back silently to the original sources and query. Everything still works, just without refinement.

**Why this matters:** Without the router, "pull up PR #123" would search Notion and Slack for the string "pull up PR #123". With the router, it searches only GitHub for "123".

### API Clients

Each client follows the same pattern:

```python
async def search_X(query: str, max_results: int = 3) -> list[IntegrationResult]:
    # call API
    # return [] on any error — never raises
```

All network calls go through `safe_get`/`safe_post` which catch every exception and return `None` on failure. Clients return `[]` if the API call fails.

Parallel detail fetching (Notion block snippets, Asana task details) uses `asyncio.gather(return_exceptions=True)` — one slow item never blocks the others.

---

## Wiring Into the Full System

Import and pass `integration_callback` to the orchestrator:

```python
from dotenv import load_dotenv
load_dotenv()  # must be called before importing integrations

from agent import QuorumOrchestrator
from integrations import integration_callback

orch = QuorumOrchestrator(
    speak_callback=your_speak_fn,
    integration_callback=integration_callback,
    action_callback=your_action_fn,
)
```

That's it. The orchestrator calls it automatically when topics are mentioned or questions need live context.

---

## New Orchestrator Behaviour

These features were added to `agent/orchestrator.py` as part of this branch.

### Clarifying Questions

If someone says "pull up the PR" with no PR number, Quorum asks instead of guessing:

```
User:  "Pull up the PR"
Quorum: "Which PR are you referring to?"
User:  "PR 42"
Quorum: "Pulling up the PR for PR 42 now."
```

- Pending state is stored per meeting, keyed by `meeting_id`
- Times out after **30 seconds** — if no answer, Quorum moves on
- Currently triggers for: `ACTION_PR` with no reference, `ACTION_TASK` with no title

### Proactive Mode

Off by default. Enable with `QUORUM_PROACTIVE=true` in `.env`.

When enabled and mode is `active`, instead of immediately creating a task, Quorum offers first:

```
User:  "Add that as a task"
Quorum: "I could create a task for that — want me to?"
User:  "Yes"
Quorum: [creates task]
```

Affirmative keywords: `yes, yeah, yep, sure, go ahead, do it, please`
Negative keywords: `no, nope, don't, skip, cancel`
Offer times out after **20 seconds**.

### Cooldown

In `active` mode, Quorum won't surface topic context more than once every **15 seconds**. Prevents it from interrupting constantly in fast-moving meetings.

### Deduplication

If the same URL has already been surfaced in a meeting, it won't be surfaced again. Tracked via `MeetingContext._surfaced` which already existed — just now checked before speaking.

### Mode Summary

| Mode | Behaviour |
|---|---|
| `on_demand` (default) | Speaks only when addressed ("Hey Quorum...") |
| `active` | Speaks proactively on topic mentions, questions, decisions |
| `active` + `QUORUM_PROACTIVE=true` | Same as active, but offers before acting on tasks/PRs |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

### Required
| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | From openrouter.ai — used for all LLM calls (Hermes 3 405B) |

### GitHub
| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | Personal access token. Needs `Pull requests: read`. Works without it for public repos at lower rate limits. |
| `GITHUB_OWNER` | Org or user name (e.g. `acme-corp`). Used for direct PR lookup by number. |
| `GITHUB_REPO` | Repo name (e.g. `backend`). Used for direct PR lookup by number. |

Without `GITHUB_OWNER`/`GITHUB_REPO`, GitHub searches all public repos — still works, just broader.

### Notion
| Variable | Description |
|---|---|
| `NOTION_TOKEN` | Integration secret from notion.so/my-integrations. Starts with `ntn_` or `secret_`. |

**Important:** After creating the integration, go to each Notion page → `···` → Connections → add your integration. Pages not connected will not appear in search results.

Notion search matches **page titles only** — not content inside pages.

### Slack
| Variable | Description |
|---|---|
| `SLACK_BOT_TOKEN` | Must be a **user token** (`xoxp-`), not a bot token (`xoxb-`). The `search:read` scope requires a user token. |

To get a user token: Slack app → OAuth & Permissions → add `search:read` under **User Token Scopes** (not Bot Token Scopes) → reinstall → copy the `xoxp-` token.

### Asana
| Variable | Description |
|---|---|
| `ASANA_TOKEN` | Personal access token from app.asana.com → avatar → My Settings → Apps |
| `ASANA_WORKSPACE_GID` | Run: `curl.exe -H "Authorization: Bearer YOUR_TOKEN" https://app.asana.com/api/1.0/workspaces` and copy the `gid` |

### Agent Behaviour
| Variable | Default | Description |
|---|---|---|
| `QUORUM_MODE` | `on_demand` | `on_demand` or `active` |
| `QUORUM_PROACTIVE` | `false` | `true` to enable offer-before-acting in active mode |
| `LOG_LEVEL` | `INFO` | Use `DEBUG` to see every decision |

---

## Running Tests

### Agent tests (offline, no keys needed)

```bash
python test_integration.py
```

Expected: 5/5 passed.

### Integration API tests (requires real keys in .env)

```bash
python test_apis.py
```

Tests each source with a known-good query and a known-bad query. Edit the queries in `test_apis.py` to match real content in your workspace.

---

## Gotchas & Watch Outs

### OpenRouter rate limits

The free tier for Hermes 3 405B is **50 requests/day**. The router makes one LLM call per integration request — in a real meeting with frequent topic mentions, you will hit this quickly. Options:

- Add credits to your OpenRouter account (removes the limit)
- Use a dedicated Hermes endpoint if the hackathon provides one
- The router falls back gracefully — everything still works without LLM routing

### `load_dotenv()` must be called before imports

The integration clients read env vars at module load time. If you call `load_dotenv()` after importing `integrations`, the tokens will be empty.

```python
# correct
from dotenv import load_dotenv
load_dotenv()
from integrations import integration_callback

# wrong — tokens won't load
from integrations import integration_callback
from dotenv import load_dotenv
load_dotenv()
```

### All API timeouts are 3 seconds

If any API is slow, it returns `[]` after 3 seconds. The meeting is never blocked. Partial results from other sources are still returned.

### `asyncio.gather(return_exceptions=True)` throughout

One failing source (e.g. Slack token missing) never crashes the whole integration call. The other sources still return results. Errors are logged at WARNING level.

### `mode_state.json` persists mode across restarts

If Quorum is unexpectedly silent or noisy, check `mode_state.json` in the working directory. Delete it to reset to `on_demand`.
