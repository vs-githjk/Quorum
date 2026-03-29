# Q — Session Notes

Session date: 2026-03-29
Branch: `voice-bot`
Commit: `34c8e13`

---

## What we fixed

### 1. Q was responding to everything (critical bug)

**Symptom:** Q spoke on every utterance — "It doesn't matter", "See?", "Thank you", "Hey, April" — even in `on_demand` mode.

**Root cause:** `QOrchestrator.process_segment()` called `_set_exchange_engaged()` unconditionally on every incoming segment. This flipped exchange state to `ENGAGED` for all speech, bypassing the `on_demand` gate in `_process_flushed`.

**Fix:** `agent/orchestrator.py:315` — only reset the idle timer if an exchange is already active:
```python
# Before
self._set_exchange_engaged(mid)

# After
if self._get_exchange_state(mid) == _EXCHANGE_ENGAGED:
    self._set_exchange_engaged(mid)
```

---

### 2. Q was narrating instead of talking to the speaker

**Symptom:** When the bug above let unaddressed speech through, Q generated commentary like *"It seems Abhinav is trying to get your attention..."* instead of staying silent or addressing the speaker directly.

**Fix:** Two rules added to `_SYSTEM_PROMPT` in `agent/q_agent.py`:
- Always address the speaker in second person — no narrating
- Respond with exactly `SKIP` when the message is not a direct request to Q (Q already handles `SKIP` as a stay-silent signal)

---

### 3. Q forgot task context across turns (short-term memory)

**Symptom:** User created a task ("And create a task for me?") then asked to update it ("create a due date for the created task"). Q searched Asana for the literal phrase "created task", found nothing, and asked the user to clarify — despite having just created the task with a known `task_gid`.

**Root cause:** `QAgent.run()` rebuilt the `messages` list from scratch on every call. The only context passed to the LLM was raw transcript text — not Q's previous responses, tool calls, or tool results. `task_gid`, URLs, and prior search results were discarded the moment `run()` returned.

**Fix:** Rolling agent exchange history in `agent/context.py` + `agent/q_agent.py`:

- `MeetingContext._agent_history` stores the last 10 full exchanges (user + tool calls + tool results + final answer) per meeting.
- `QAgent.run()` injects the last 3 exchanges into the `messages` list between the system prompt and the current user message before calling the LLM.
- After responding, `run()` saves the completed exchange back to context.

Now if Q creates "Task for Abhinav" (gid: `1213865...`), the next turn sees that in its message history and can call `update_asana_task` directly without re-searching.

**Key methods added:**
- `MeetingContext.add_agent_exchange(meeting_id, exchange)` — save a turn
- `MeetingContext.get_agent_history(meeting_id, n=3)` — retrieve for injection

---

### 4. Cross-meeting memory for tool actions

**What existed:** At meeting end, a 3-sentence LLM summary + decisions + 500-char transcript snippet were persisted to `meeting_history.json`. The `search_past_meetings` tool let Q query these.

**What was missing:** Tool actions (tasks created, tasks updated, Slack searches, etc.) were not captured. So "what tasks did we create last week?" would return nothing useful.

**Fix:** `add_agent_exchange` now also scans each tool result message and extracts a one-liner summary (e.g. `[create_asana_task] Created task: Task for Abhinav (https://...)`). These are stored in `MeetingContext._tool_actions` and persisted under `tool_actions` in `meeting_history.json` at meeting end. `search_past_meetings` now searches and displays this field.

---

### 5. Rename: Quorum → Q (pre-existing, included in this commit)

`agent/mode.py`, `agent/intent.py`, `voice/speak.py` had already been updated to use `Q_TRIGGERS = ["q", "hey q"]` and rename `QuorumSpeaker` → `QSpeaker` and `is_addressed_to_quorum` → `is_addressed_to_q`. These were staged and included in the same commit.

---

## Files changed

| File | What changed |
|------|-------------|
| `agent/orchestrator.py` | Fix unconditional exchange state set |
| `agent/q_agent.py` | SKIP + second-person prompt rules; inject + save history |
| `agent/context.py` | `_agent_history`, `_tool_actions`, `add_agent_exchange`, `get_agent_history`; persist `tool_actions` to meeting history |
| `agent/mode.py` | Rename Quorum → Q triggers; word-boundary matching for "q" |
| `agent/intent.py` | Rename trigger constant and method |
| `voice/speak.py` | Rename `QuorumSpeaker` → `QSpeaker` |

---

## Known issues still open

- `send_chat_message` — LLM doesn't always call it even when explicitly asked. Needs prompt tuning or a post-processing hook.
- No `complete_asana_task` tool — "mark the task done" has no handler. Needs `PUT /tasks/{gid}` with `{"data": {"completed": true}}`.
- Assignee by name — `update_asana_task` supports `"me"` but not other people's names. Needs a `/users` lookup.
- Hermes always unavailable — every LLM call logs a warning before falling back to OpenRouter. Suppress or remove the Hermes attempt if not running locally.
- Exchange goes IDLE mid-sentence — 20s timeout can expire during a slow multi-part request ("I would like to... of... can you..."), dropping the follow-up. Could increase timeout or reset on Q's own speech.
