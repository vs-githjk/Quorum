# Quorum — Agent Branch: Integration Guide

This document covers everything built in the `agent` branch and exactly what each teammate needs to do to integrate with it.

---

## Table of Contents

1. [What Was Built](#what-was-built)
2. [File Overview](#file-overview)
3. [Setup](#setup)
4. [How the Agent Works](#how-the-agent-works)
5. [Interface Contracts](#interface-contracts)
6. [Voice-Bot Integration](#voice-bot-integration-recall--deepgram--elevenlabs)
7. [Integrations Integration](#integrations-integration-github--notion--slack--asana)
8. [Actions-UI Integration](#actions-ui-integration)
9. [Running Tests](#running-tests)
10. [Environment Variables](#environment-variables)
11. [Gotchas & Watch Outs](#gotchas--watch-outs)

---

## What Was Built

The `agent/` directory is the complete intelligence layer for Quorum. It receives live transcript text, decides what (if anything) to do, and dispatches to the other three branches.

Five files were written:

| File | Role |
|---|---|
| `agent/mode.py` | Controls whether Quorum speaks proactively or only on-demand |
| `agent/intent.py` | Classifies every transcript segment into an intent type |
| `agent/context.py` | Meeting memory — stores transcript, decisions, actions; searches past meetings |
| `agent/orchestrator.py` | The main loop — wires everything together, calls LLM, dispatches callbacks |
| `agent/__init__.py` | Clean public API — import everything from here |

Supporting files:

| File | Role |
|---|---|
| `requirements.txt` | Python dependencies (`aiohttp` only) |
| `.env.example` | Template for required environment variables |
| `test_integration.py` | End-to-end test — runs fully offline with mock callbacks |

---

## File Overview

### `agent/mode.py` — ModeManager

Two modes:

- **`on_demand`** (default): Quorum only speaks when someone says "Quorum" or "hey Quorum" in the transcript.
- **`active`**: Quorum speaks proactively whenever it detects something relevant.

Mode is persisted to `mode_state.json` on disk so it survives restarts. Default is `on_demand` — safe for demos where you don't want it speaking unexpectedly.

---

### `agent/intent.py` — IntentDetector

Classifies transcript text into one of 7 intent types using keyword matching (fast, no API call). Only escalates to LLM when the match is ambiguous.

| Intent | Example trigger |
|---|---|
| `ACTION_TASK` | "Add that as a task", "create a ticket" |
| `ACTION_PR` | "Pull up the PR", "show me the pull request" |
| `ACTION_CHART` | "Show me a chart", "can you visualize" |
| `DECISION` | "We'll go with", "agreed", "decided" |
| `QUESTION` | "What did we", "what's the status", "can you find" |
| `TOPIC_MENTION` | Capitalized proper nouns OR "let's talk about X" |
| `NONE` | Everything else — Quorum stays silent |

Detection priority runs top to bottom in that table. Action intents always win over softer signals.

---

### `agent/context.py` — MeetingContext

Stores everything during a meeting (in memory) and persists to `meeting_history.json` when the meeting ends.

- Full transcript history per meeting
- Decisions logged when a `DECISION` intent fires
- Actions dispatched during the meeting
- Integration results that were surfaced

Cross-meeting search uses keyword overlap — no vector DB, no embeddings. Fast and fully offline.

---

### `agent/orchestrator.py` — QuorumOrchestrator

The heart. For each final transcript segment:

```
receive segment
  → skip if not final
  → store in context
  → detect intent
  → if NONE: stay quiet
  → check mode (active vs on_demand)
  → if mode says no: log and return
  → dispatch to the right handler
```

LLM calls try Hermes locally first (2s timeout), then fall back to OpenRouter if Hermes is down or slow.

---

## Setup

```bash
git clone https://github.com/vs-githjk/Quorum.git
cd Quorum
git checkout agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in your OPENROUTER_API_KEY in .env
```

---

## How the Agent Works

```
voice-bot
  │
  │  TranscriptSegment (every Deepgram final event)
  ▼
QuorumOrchestrator.process_segment()
  │
  ├─► ModeManager.should_respond()
  │       on_demand: only if "quorum" mentioned
  │       active:    if intent != NONE
  │
  ├─► IntentDetector.detect()
  │       keyword match → Intent(type, confidence, topic, requires_llm)
  │
  ├─── if ACTION_TASK / ACTION_PR / ACTION_CHART
  │       └─► action_callback(ActionRequest)       → actions-ui
  │
  ├─── if DECISION
  │       └─► context.add_decision()
  │           speak_callback(SpeakCommand)          → voice-bot / ElevenLabs
  │
  ├─── if QUESTION
  │       └─► context.search_past_meetings()
  │           [if needs_llm] integration_callback() → integrations
  │           call_llm()
  │           speak_callback(SpeakCommand)          → voice-bot / ElevenLabs
  │
  └─── if TOPIC_MENTION
          └─► integration_callback(ContextRequest)  → integrations
              call_llm() → decide if worth surfacing
              speak_callback(SpeakCommand)          → voice-bot / ElevenLabs
```

---

## Interface Contracts

All dataclasses are defined in `agent/orchestrator.py` and re-exported from `agent/__init__.py`. **Import only from `agent`** — never from sub-modules directly.

```python
from agent import (
    TranscriptSegment,
    SpeakCommand,
    ContextRequest,
    IntegrationResult,
    ActionRequest,
    QuorumOrchestrator,
)
```

### TranscriptSegment — voice-bot → agent

```python
@dataclass
class TranscriptSegment:
    text: str         # the spoken words
    speaker: str      # "Speaker 0", "Speaker 1", etc. from Deepgram diarization
    timestamp: float  # unix timestamp
    is_final: bool    # IMPORTANT: agent ignores False — only send True when Deepgram commits
    meeting_id: str   # unique ID for this meeting session, consistent across the whole call
```

### SpeakCommand — agent → voice-bot

```python
@dataclass
class SpeakCommand:
    text: str         # what Quorum says aloud — pass directly to ElevenLabs
    meeting_id: str
    priority: str     # "high" = interrupt; "normal" = wait for silence
```

### ContextRequest — agent → integrations

```python
@dataclass
class ContextRequest:
    query: str        # search term extracted from transcript
    sources: list[str]  # e.g. ["github", "notion", "slack"] — only query what's listed
    meeting_id: str
    max_results: int  # default 3 — cap results per source to this
```

### IntegrationResult — integrations → agent

```python
@dataclass
class IntegrationResult:
    source: str       # "github" | "notion" | "slack" | "asana"
    title: str        # human-readable title
    url: str          # direct link
    summary: str      # 1–2 sentence description — this is what the LLM sees
    raw_data: dict    # full payload for actions-ui to render
    timestamp: float  # unix timestamp
```

### ActionRequest — agent → actions-ui

```python
@dataclass
class ActionRequest:
    action_type: str  # "create_task" | "pull_pr" | "generate_chart"
    parameters: dict  # action-specific — see table below
    context: str      # the transcript text that triggered this
    meeting_id: str
```

`parameters` contents by action type:

| `action_type` | `parameters` keys |
|---|---|
| `create_task` | `title` (str), `source_text` (str) |
| `pull_pr` | `reference` (str — PR number or name), `source_text` (str) |
| `generate_chart` | `topic` (str), `source_text` (str) |

---

## Voice-Bot Integration (Recall + Deepgram + ElevenLabs)

### What you need to do

1. Copy the dataclasses from `agent` — do not redefine them.
2. Instantiate the orchestrator at the start of a meeting session.
3. On every Deepgram transcript event, build a `TranscriptSegment` and call `process_segment`.
4. Implement `speak_callback` to pass `SpeakCommand.text` to ElevenLabs.

### Minimal wiring example

```python
from agent import QuorumOrchestrator, TranscriptSegment, SpeakCommand

async def speak_callback(cmd: SpeakCommand):
    # cmd.priority == "high" → interrupt; "normal" → queue
    await elevenlabs_speak(cmd.text)

orch = QuorumOrchestrator(
    speak_callback=speak_callback,
    integration_callback=your_integration_fn,   # from integrations branch
    action_callback=your_action_fn,             # from actions-ui branch
)

await orch.start_meeting(meeting_id)

# On every Deepgram event:
async def on_deepgram_event(event):
    seg = TranscriptSegment(
        text=event["channel"]["alternatives"][0]["transcript"],
        speaker=event.get("speaker", "Speaker 0"),
        timestamp=time.time(),
        is_final=event["is_final"],   # ← Deepgram sets this
        meeting_id=meeting_id,
    )
    await orch.process_segment(seg)
```

### Watch outs

- **Always pass `is_final` correctly.** The agent drops all non-final segments. If you always pass `True`, it will fire on every word fragment and spam callbacks.
- **`meeting_id` must be consistent** for the entire call. Use the Recall bot's session ID or generate a UUID at call start. If it changes mid-call, context is split across two meetings.
- **Do not await `process_segment` in Deepgram's callback directly** if your callback is sync — wrap it: `asyncio.create_task(orch.process_segment(seg))`.

---

## Integrations Integration (GitHub, Notion, Slack, Asana)

### What you need to do

Implement one async function with this exact signature:

```python
async def integration_callback(req: ContextRequest) -> list[IntegrationResult]:
    results = []
    for source in req.sources:
        if source == "github":
            results += await search_github(req.query, req.max_results)
        elif source == "notion":
            results += await search_notion(req.query, req.max_results)
        elif source == "slack":
            results += await search_slack(req.query, req.max_results)
        elif source == "asana":
            results += await search_asana(req.query, req.max_results)
    return results
```

Each item in the returned list must be an `IntegrationResult`. The agent uses `title` and `summary` to build its spoken response — keep `summary` to 1–2 sentences.

### Watch outs

- **Return an empty list `[]` if nothing is found** — never return `None`. The agent checks `if not results` before acting.
- **`summary` is what gets read aloud.** Keep it short and natural-language. The agent passes it directly to the LLM prompt.
- **`raw_data` is for actions-ui**, not the agent — put the full API payload there so the dashboard can render it without a second API call.
- **Respect `max_results`.** The agent caps it at 3 by default. Returning 20 results means 20 items go into the LLM prompt — it will slow down and cost more tokens.
- **Don't block the event loop.** All your API calls must be `async`. Use `aiohttp` (already in requirements) or `httpx`.

---

## Actions-UI Integration

### What you need to do

Implement one async function with this exact signature:

```python
async def action_callback(req: ActionRequest) -> dict:
    if req.action_type == "create_task":
        return await create_asana_task(req.parameters)
    elif req.action_type == "pull_pr":
        return await fetch_github_pr(req.parameters)
    elif req.action_type == "generate_chart":
        return await render_chart(req.parameters)
    return {"status": "unknown_action"}
```

The return value is a plain dict. The agent logs it but does not depend on its shape — you can put whatever you want in there for your own use.

### Watch outs

- **The agent already spoke a confirmation before your callback returns.** e.g. "Got it — I've added that as a task." Do not have the agent speak again for the same action — your callback return value does not trigger TTS.
- **`req.context` is the raw transcript text that triggered the action.** Use it for task descriptions, PR search terms, or chart titles.
- **`req.parameters["source_text"]` is always set.** The other keys depend on `action_type` — see the parameters table above.

---

## Running Tests

### Individual module tests (each file has its own test)

```bash
source venv/bin/activate

python3 -m agent.mode          # 9 tests
python3 -m agent.intent        # 10 tests
python3 -m agent.context       # 15 tests
python3 -m agent.orchestrator  # 5 tests
```

### Full end-to-end integration test

```bash
python3 test_integration.py
```

Expected output:

```
======================================================================
  Quorum Agent — Final Integration Test
======================================================================

  [PASS] #1 "Let's talk about the new authentication feature"
         Expected : topic surfaced
         Got      : integration query='the new authentication feature'

  [PASS] #2 "We'll go with OAuth 2.0, everyone agreed"
         Expected : decision logged
         Got      : decision='OAuth 2.0, everyone agreed'

  [PASS] #3 'Hey Quorum, what did we decide last week about t'
         Expected : past meeting search + spoke
         Got      : spoke='Last week the team decided to use OAuth 2.0...'

  [PASS] #4 'Add that as a task for the backend team'
         Expected : action_callback: create_task
         Got      : action_type='create_task'

  [PASS] #5 'Can you show me a chart of our sprint velocity?'
         Expected : action_callback: generate_chart
         Got      : action_type='generate_chart'

──────────────────────────────────────────────────────────────────────
  Result: 5/5 passed
──────────────────────────────────────────────────────────────────────
```

The test runs fully offline — no Hermes, no OpenRouter, no real API keys needed.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes (for LLM) | — | Get from openrouter.ai. Used as LLM fallback when Hermes is down. |
| `HERMES_HOST` | No | `http://localhost:11434` | URL of your local Ollama/Hermes instance. |
| `QUORUM_MODE` | No | `on_demand` | Starting mode. Set to `active` for demos where you want it proactive. |
| `LOG_LEVEL` | No | `INFO` | Python logging level. Use `DEBUG` to see every decision the agent makes. |

The agent never hardcodes keys. All secrets come from `os.getenv()`.

---

## Gotchas & Watch Outs

### For everyone

- **Import from `agent`, not from `agent.orchestrator` directly.** The `__init__.py` is the stable public API. Sub-module structure may change.
- **`venv/` is gitignored.** Each teammate runs `pip install -r requirements.txt` in their own venv.
- **`mode_state.json` and `meeting_history.json` are runtime files** created at the working directory. Do not commit them. They are already in `.gitignore`.
- **The agent only runs on Python 3.11+.** The `str | None` union syntax requires it.

### LLM behaviour

- Hermes is tried first with a **2-second timeout**. If your local Ollama is slow to load the model on first call, the first request will fall back to OpenRouter. This is expected — subsequent calls will hit Hermes.
- If both Hermes and OpenRouter fail (no internet, no key), the agent returns the string `"I wasn't able to retrieve that information right now."` and continues — it does not crash.
- The LLM system prompt keeps responses to 2 sentences max. If you find Quorum being too verbose in the meeting, this is where to tighten it.

### Mode behaviour

- Default mode is **`on_demand`**. In this mode, Quorum will not speak unless someone says "Quorum" or "hey Quorum" in the transcript. This is intentional — you do not want it interrupting during a real demo.
- Switch to **`active`** mode for testing or if you want proactive behaviour during the demo: `orch._mode.set_mode("active")` or set `QUORUM_MODE=active` and call `orch._mode.set_mode(os.getenv("QUORUM_MODE", "on_demand"))` at startup.
- Mode persists across restarts via `mode_state.json`. If Quorum is being unexpectedly silent or unexpectedly noisy, check that file first.

### Timing

- `process_segment` is async but not parallelised per meeting. If a Deepgram event arrives while a previous segment is still being processed (e.g. waiting on an integration API), they will queue. For the hackathon this is fine. If you hit latency issues, wrap calls in `asyncio.create_task`.
- Integration and action callbacks must complete before the agent speaks. Keep your API calls fast — aim for under 1 second. If a source is slow, time it out and return partial results.
