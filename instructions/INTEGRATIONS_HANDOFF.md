# Quorum — Voice-Bot & Agent: Integrations Handoff

Everything that has been built and is running on the `voice-bot` branch. Read this before touching anything.

---

## Table of Contents

1. [What's Been Built](#whats-been-built)
2. [How the Full System Works](#how-the-full-system-works)
3. [Your One Job](#your-one-job)
4. [The Exact Interface Contract](#the-exact-interface-contract)
5. [Running the Bot Locally](#running-the-bot-locally)
6. [File Overview](#file-overview)
7. [Environment Variables](#environment-variables)
8. [What's Stubbed and Where](#whats-stubbed-and-where)
9. [LLM Behaviour](#llm-behaviour)
10. [Trigger Words & Mode](#trigger-words--mode)
11. [Gotchas & Watch Outs](#gotchas--watch-outs)

---

## What's Been Built

The `voice-bot` branch contains the full running bot. It:

1. Joins a Google Meet / Zoom / Teams meeting via **Recall.ai**
2. Receives live transcripts over WebSocket (Deepgram Nova-2 via Recall)
3. Runs every transcript segment through the **agent intelligence layer** (intent detection, context, LLM)
4. Generates spoken responses via **ElevenLabs TTS**
5. Injects the audio back into the meeting through the bot's microphone via **Recall.ai audio injection**

Everything works end-to-end. The only missing piece is **real integration data** — the integrations callback currently returns an empty list, which is where you come in.

---

## How the Full System Works

```
You say something in the meeting
        │
        ▼
Recall.ai bot (sitting in the call)
        │  streams transcript over WebSocket
        ▼
/ws/transcript endpoint (FastAPI server, port 8000, tunnelled via ngrok)
        │
        ▼
AudioStreamManager  →  on_transcript_segment()
        │
        ▼
QuorumOrchestrator.process_segment()
        │
        ├─ [skip if not final]
        ├─ store in MeetingContext
        ├─ IntentDetector.detect() → intent type
        │
        ├─ if NONE + "hey q" said → acknowledge ("I'm here. What do you need?")
        │
        ├─ if TOPIC_MENTION → integration_callback(ContextRequest) ← YOUR CODE
        │       └─ if results → LLM → speak
        │       └─ if no results + "hey q" said → acknowledge
        │
        ├─ if QUESTION → search past meetings + integration_callback ← YOUR CODE
        │       └─ LLM → speak
        │
        ├─ if ACTION_TASK / ACTION_PR / ACTION_CHART → action_callback (stub)
        │       └─ speak confirmation
        │
        └─ if DECISION → log decision → speak confirmation
                │
                ▼
        QuorumSpeaker.speak(text)
                │  ElevenLabs API → MP3 bytes
                ▼
        RecallClient.inject_audio()
                │  POST base64 MP3 to Recall.ai /bot/{id}/output_audio/
                ▼
        Bot speaks in the meeting
```

---

## Your One Job

Implement the `integration_callback`. It currently lives as a stub in `main_bot.py`:

```python
async def _integration_stub(self, req: ContextRequest) -> list[IntegrationResult]:
    logger.info("Integration request (stub): query=%r sources=%s", req.query, req.sources)
    return []
```

Replace this with a real function that queries GitHub, Notion, Slack, Asana — whatever sources your team has set up — and returns a list of `IntegrationResult` objects.

That's it. You don't touch the agent, the voice layer, or the bot layer. Just implement that one function and wire it in.

---

## The Exact Interface Contract

Import everything from `agent` — never from sub-modules directly:

```python
from agent import ContextRequest, IntegrationResult
```

### ContextRequest — what the agent sends you

```python
@dataclass
class ContextRequest:
    query: str          # search term pulled from the transcript, e.g. "authentication feature"
    sources: list[str]  # which integrations to hit, e.g. ["github", "notion", "slack"]
    meeting_id: str     # current meeting session ID
    max_results: int    # default 3 — cap your results per source to this number
```

`sources` is populated by the agent based on intent:
- `TOPIC_MENTION` → `["github", "notion", "slack"]`
- `QUESTION` → `["notion", "slack"]`

Only query the sources listed. Don't query GitHub if it's not in the list.

### IntegrationResult — what you return

```python
@dataclass
class IntegrationResult:
    source: str       # "github" | "notion" | "slack" | "asana"
    title: str        # human-readable title shown in logs
    url: str          # direct link to the resource
    summary: str      # 1–2 sentences — THIS IS WHAT GETS READ ALOUD by the LLM
    raw_data: dict    # full API payload — for the dashboard/UI to render later
    timestamp: float  # unix timestamp of the result (when it was created or fetched)
```

**`summary` is the most important field.** The agent passes it directly into the LLM prompt. Keep it short, factual, and in plain English. The LLM then decides whether it's worth surfacing and reformulates it into a spoken sentence.

### Your function signature

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

Return `[]` if nothing is found. Never return `None`. The agent checks `if not results` — returning `None` will crash it.

### Wiring it into main_bot.py

Find `_integration_stub` in `main_bot.py` and replace it:

```python
# Before (stub):
async def _integration_stub(self, req: ContextRequest) -> list[IntegrationResult]:
    return []

# After (real):
async def _integration_stub(self, req: ContextRequest) -> list[IntegrationResult]:
    return await your_integration_module.fetch(req)
```

The orchestrator is already wired to call `self._integration_stub` — you just need to fill it in. No other file needs to change.

---

## Running the Bot Locally

### Prerequisites

- Python 3.11+
- ngrok installed (`brew install ngrok`)
- A `.env` file with all required keys (see [Environment Variables](#environment-variables))

### Steps

```bash
# 1. Clone and set up
git clone https://github.com/vs-githjk/Quorum.git
cd Quorum
git checkout voice-bot

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in all required keys in .env

# 2. Start ngrok in a separate terminal (keep it running)
ngrok http 8000
# Copy the https://... URL it gives you and set WEBHOOK_BASE_URL in .env

# 3. Run the bot
python3 main_bot.py
# Paste a Google Meet / Zoom / Teams URL when prompted
# Press Ctrl+C to stop
```

### Verify it's working

Before joining a meeting, confirm ngrok is tunnelling correctly:
```bash
curl https://your-ngrok-url.ngrok-free.app/health
# Expected: {"status":"ok","bot_active":false,"meeting_id":null,"segments_received":0}
```

### Test TTS in isolation (no meeting needed)

```bash
python3 -m voice.speak
# Generates test_output.mp3 — confirm ElevenLabs is working
```

### Run integration tests (fully offline, no API keys needed)

```bash
python3 test_integration.py
# Expected: 5/5 passed
```

---

## File Overview

```
Quorum/
├── main_bot.py              ← Entry point. Wire your integration_callback here.
├── requirements.txt
├── .env                     ← All secrets (gitignored)
├── .env.example             ← Template
│
├── agent/                   ← Intelligence layer (don't modify)
│   ├── __init__.py          ← Public API — import from here
│   ├── orchestrator.py      ← Main decision loop + all dataclasses
│   ├── intent.py            ← Keyword-based intent classification
│   ├── context.py           ← Meeting memory + cross-meeting search
│   └── mode.py              ← on_demand vs active mode
│
├── bot/                     ← Recall.ai integration (don't modify)
│   ├── __init__.py          ← BotStatus, TranscriptSegment dataclasses
│   ├── recall_client.py     ← Recall.ai REST API client
│   └── audio_stream.py      ← FastAPI server + WebSocket handler
│
├── voice/                   ← Speech layer (don't modify)
│   ├── speak.py             ← ElevenLabs TTS queue
│   └── transcribe.py        ← Deepgram streaming transcriber
│
└── instructions/
    ├── README.md            ← Agent branch guide (older)
    └── INTEGRATIONS_HANDOFF.md  ← This file
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `RECALL_API_KEY` | Yes | Recall.ai API token. Get from recall.ai dashboard. |
| `DEEPGRAM_API_KEY` | Yes | Deepgram API key. Used by Recall.ai for transcription. |
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs API key for TTS. |
| `ELEVENLABS_VOICE_ID` | Yes | ElevenLabs voice ID. Current: `uju3wxzG5OhpWcoi3SMy`. |
| `WEBHOOK_BASE_URL` | Yes | Your ngrok public URL. Must be live when the bot is running. |
| `OPENROUTER_API_KEY` | Yes (for LLM) | OpenRouter API key. Free tier available at openrouter.ai. |
| `HERMES_HOST` | No | Local Ollama URL. Default: `http://localhost:11434`. |
| `BOT_NAME` | No | Display name in meeting. Default: `Quorum`. |
| `SERVER_PORT` | No | Local server port. Default: `8000`. |
| `QUORUM_MODE` | No | Starting mode: `on_demand` or `active`. Default: `on_demand`. |
| `LOG_LEVEL` | No | Logging verbosity. Default: `INFO`. Use `DEBUG` to see every decision. |

**The bot will refuse to start if any of the first 5 are missing.**

---

## What's Stubbed and Where

| Location | What's stubbed | What you replace it with |
|---|---|---|
| `main_bot.py:_integration_stub` | Returns `[]` always | Real GitHub/Notion/Slack/Asana queries |
| `main_bot.py:_action_stub` | Logs and returns `{"status": "stub"}` | Actions-UI teammate handles this |

Both stubs are in the `QuorumBot` class in `main_bot.py`. They're clearly marked with `# stub` in the docstrings.

---

## LLM Behaviour

The agent uses two LLMs in priority order:

1. **Hermes 3 (local Ollama)** — tried first, 2-second timeout. If you have Ollama running locally with the `hermes3` model, set `HERMES_HOST=http://localhost:11434`. It's faster and free.
2. **OpenRouter (Hermes-3-405B)** — fallback if local Hermes is down or slow. Requires `OPENROUTER_API_KEY`. Free tier has rate limits — if you hit a `429`, wait a minute.

If both fail, the agent says `"I wasn't able to retrieve that information right now."` and continues. It does not crash.

**The LLM is responsible for deciding whether integration results are worth surfacing.** Even if you return results, the LLM might say `SKIP` if it doesn't think they're relevant to the current conversation. This is intentional — it prevents Quorum from being noisy. If you're testing and want to confirm your results are coming through, check the logs:

```
INFO  agent.orchestrator — [mtg-id] Topic mentioned: 'auth' — fetching context
INFO  __main__ — Integration request: query='auth' sources=['github', 'notion', 'slack']
```

---

## Trigger Words & Mode

Quorum responds to these spoken phrases (case-insensitive, Deepgram transcription variants included):

```
"quorum"      "hey quorum"
"hey q"
"coram"       "hey coram"      ← Deepgram mishears "quorum" as these
"korem"       "hey korem"
```

There are two modes, persisted in `mode_state.json`:

- **`on_demand`** (safe default): Quorum only speaks when one of the trigger words above is detected OR when a recognized intent keyword fires AND the trigger is present.
- **`active`**: Quorum speaks proactively on any recognized intent, even without a trigger word. Good for demos.

Current mode is stored in `mode_state.json` at the project root. Delete this file to reset to `on_demand`.

---

## Gotchas & Watch Outs

### ngrok must be running before starting the bot

The bot registers its webhook URL with Recall.ai at join time. If ngrok isn't running or the URL in `.env` is stale, Recall.ai can't connect and you'll get no transcripts. Always start ngrok first, copy the URL to `.env`, then start the bot.

### Deepgram transcribes "Quorum" inconsistently

In real meetings it comes through as "Quorum", "Coram", "Korem", "Corem". All variants are handled in `QUORUM_TRIGGERS` in `agent/mode.py`. If you see a new variant in the logs, add it there.

### `summary` in IntegrationResult is what the LLM sees

The LLM is given your `summary` field verbatim. If it's too long, vague, or technical, the LLM will either say `SKIP` (won't surface it) or give a bad spoken response. Target 1–2 clear, conversational sentences like `"The PR for the auth refactor was merged last Thursday by @alice. It switches all API routes to OAuth 2.0."`

### Return empty list, not None

The agent does `if not results: return`. Returning `None` instead of `[]` will raise a `TypeError` on `for r in results`. Always return a list.

### OpenRouter free tier has rate limits

The free Hermes-3-405B model on OpenRouter hits 429s if you make too many calls in a short window. For testing, use local Ollama (`HERMES_HOST=http://localhost:11434`) to avoid this. The paid tier doesn't have this issue.

### Don't block the event loop

All your integration API calls must be `async`. The bot runs in a single asyncio event loop. A blocking `requests.get()` inside your callback will freeze transcription. Use `aiohttp` (already in `requirements.txt`) or `httpx`.

### `raw_data` is for the UI, not the agent

Put the full API response in `raw_data`. The agent ignores it — it only uses `title`, `summary`, `source`, and `url`. The actions-UI teammate will read `raw_data` to render cards in the dashboard.

### `mode_state.json` and `meeting_history.json` are runtime files

Do not commit them. They're gitignored. `meeting_history.json` persists context across meetings — delete it if you want a clean slate between test sessions.

### Python 3.11+ required

The codebase uses `str | None` union syntax and other 3.11 features. Check with `python3 --version` before running.
