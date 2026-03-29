import asyncio, os, re, json
import aiohttp
from dotenv import load_dotenv
load_dotenv()

from agent.context import MeetingContext
from agent.q_agent import QAgent
from integrations.asana import search_asana, create_asana_task
from integrations.slack import search_slack
from integrations.notion import search_notion
from integrations.github import search_github

_SCREEN_API_URL   = os.getenv("SCREEN_API_URL", "http://localhost:5000")
_QUORUM_MODE      = os.getenv("QUORUM_MODE", "on_demand")
_OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
_OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
_screen_link_sent = set()

def fmt(results):
    if not results:
        return "No results found."
    return "\n".join(f"- [{r.source}] {r.title}: {r.summary} ({r.url})" for r in results)

def _novnc_url():
    base = _SCREEN_API_URL.rsplit(":", 1)[0]
    return f"{base}:6080/vnc.html?autoconnect=1&resize=scale&view_only=0"

def _print_novnc_link(meeting_id):
    if meeting_id not in _screen_link_sent:
        _screen_link_sent.add(meeting_id)
        print(f"[screen] noVNC: {_novnc_url()}")

async def main():
    ctx = MeetingContext(history_file="/dev/null")
    agent = QAgent(context=ctx)

    async def tool_search_slack(meeting_id, query): return fmt(await search_slack(query))
    async def tool_search_notion(meeting_id, query): return fmt(await search_notion(query))
    async def tool_search_github(meeting_id, query): return fmt(await search_github(query))
    async def tool_search_asana(meeting_id, query): return fmt(await search_asana(query))
    async def tool_create_asana_task(meeting_id, title, notes=""): return await create_asana_task(title, notes)

    async def tool_open_on_screen(meeting_id, url):
        print(f"[tool] open_on_screen: {url}")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{_SCREEN_API_URL}/open", json={"url": url}) as r:
                body = await r.text()
                if r.status != 200:
                    return f"Error: screen returned {r.status} — {body}"
        _print_novnc_link(meeting_id)
        return f"Opened {url} on screen. Visible at: {_novnc_url()}"

    async def tool_act_on_screen(meeting_id, instruction):
        print(f"[tool] act_on_screen: {instruction}")
        summary = "Done."
        try:
            timeout = aiohttp.ClientTimeout(total=300, connect=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{_SCREEN_API_URL}/act",
                    json={"instruction": instruction},
                ) as resp:
                    _print_novnc_link(meeting_id)
                    async for line_bytes in resp.content:
                        line = line_bytes.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            event = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        etype = event.get("type")
                        if etype == "step":
                            desc = event.get("description", "")
                            if _QUORUM_MODE == "active":
                                print(f"Q [acting]: {desc}")
                            else:
                                print(f"  → {desc}")
                        elif etype == "done":
                            summary = event.get("summary", "Done.")
                            break
                        elif etype == "error":
                            summary = event.get("message") or event.get("summary", "Unknown error.")
                            print(f"  [act error] {summary}")
                            break
                        elif etype == "cancelled":
                            summary = "Task cancelled."
                            break
        except Exception as e:
            print(f"  [act_on_screen exception] {type(e).__name__}: {e}")
            return f"Error: {e}"
        return summary

    async def tool_render_visualization(meeting_id, description, data=""):
        print(f"[tool] render_visualization: {description}")
        if not _OPENROUTER_KEY:
            return "Error: OPENROUTER_API_KEY not set."
        prompt = (
            f"Generate a beautiful self-contained single-file HTML visualization.\n"
            f"Description: {description}\n"
            f"Data: {data}\n\n"
            f"Requirements:\n"
            f"- Use Chart.js from https://cdn.jsdelivr.net/npm/chart.js\n"
            f"- Dark background (#0f0f0f), vibrant colours, clean professional look\n"
            f"- Title at top, responsive layout\n"
            f"- Return ONLY the complete HTML document, no explanation"
        )
        headers = {"Authorization": f"Bearer {_OPENROUTER_KEY}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL,
                json={"model": _OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]},
                headers=headers,
            ) as r:
                resp_data = await r.json()
        html = resp_data["choices"][0]["message"]["content"]
        html = re.sub(r"^```(?:html)?\s*", "", html.strip())
        html = re.sub(r"\s*```$", "", html)
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{_SCREEN_API_URL}/render_html", json={"html": html}) as r:
                if r.status != 200:
                    body = await r.text()
                    return f"Error rendering HTML: {r.status} — {body}"
        _print_novnc_link(meeting_id)
        novnc = _novnc_url()
        return f"Visualization rendered on screen and is now visible at {novnc} — it is already open, no further action needed."

    async def tool_log_decision(meeting_id, decision):
        ctx.add_decision(decision, meeting_id); return f"Logged: {decision}"
    async def tool_search_past_meetings(meeting_id, query): return ctx.search_past_meetings(query)

    async def tool_draft_email(meeting_id, to, subject, body):
        instruction = (
            f"Open Gmail (mail.google.com) and compose a new email. "
            f"Fill in: To: {to}, Subject: {subject}, Body: {body}. "
            f"Leave it open as a draft, do not send."
        )
        return await tool_act_on_screen(meeting_id=meeting_id, instruction=instruction)

    async def tool_create_calendar_event(meeting_id, title, date, time, guests=""):
        guest_part = f" Add guests: {guests}." if guests else ""
        instruction = (
            f"Open Google Calendar (calendar.google.com) and create a new event. "
            f"Title: {title}. Date: {date}. Time: {time}.{guest_part} "
            f"Save the event."
        )
        return await tool_act_on_screen(meeting_id=meeting_id, instruction=instruction)

    async def tool_summarize_meeting(meeting_id, focus=""):
        if not _OPENROUTER_KEY:
            return "Error: OPENROUTER_API_KEY not set."
        transcript = ctx.get_recent_transcript(meeting_id, n=50)
        decisions  = ctx.get_decisions(meeting_id)
        focus_line = f"\nFocus on: {focus}" if focus else ""
        prompt = (
            f"You are summarizing a live meeting. Produce a concise summary with three sections:\n"
            f"**Decisions made**, **Action items**, **Key discussion points**.{focus_line}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"Decisions logged:\n{chr(10).join(decisions) if decisions else 'None'}"
        )
        headers = {"Authorization": f"Bearer {_OPENROUTER_KEY}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL,
                json={"model": _OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]},
                headers=headers,
            ) as r:
                data = await r.json()
        return data["choices"][0]["message"]["content"].strip()

    async def tool_ask_claude(meeting_id, prompt):
        if not _OPENROUTER_KEY:
            return "Error: OPENROUTER_API_KEY not set."
        transcript = ctx.get_recent_transcript(meeting_id, n=10)
        system = (
            "You are Q, an AI assistant embedded in a live meeting. "
            "Answer concisely and helpfully. You have context from the current meeting."
        )
        messages = [
            {"role": "system", "content": f"{system}\n\nRecent meeting context:\n{transcript}"},
            {"role": "user", "content": prompt},
        ]
        headers = {"Authorization": f"Bearer {_OPENROUTER_KEY}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL,
                json={"model": _OPENROUTER_MODEL, "messages": messages},
                headers=headers,
            ) as r:
                data = await r.json()
        return data["choices"][0]["message"]["content"].strip()

    agent.register_tools({
        "search_slack":          tool_search_slack,
        "search_notion":         tool_search_notion,
        "search_github":         tool_search_github,
        "search_asana":          tool_search_asana,
        "create_asana_task":     tool_create_asana_task,
        "log_decision":          tool_log_decision,
        "search_past_meetings":  tool_search_past_meetings,
        "open_on_screen":        tool_open_on_screen,
        "act_on_screen":         tool_act_on_screen,
        "render_visualization":  tool_render_visualization,
        "draft_email":           tool_draft_email,
        "create_calendar_event": tool_create_calendar_event,
        "summarize_meeting":     tool_summarize_meeting,
        "ask_claude":            tool_ask_claude,
    })

    mid = "test-meeting"
    print("Type a message to Q (Ctrl+C to quit)\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not text:
            continue
        response = await agent.run(text, mid, "You")
        if response:
            print(f"Q: {response.spoken}")
            if response.chat:
                print(f"   [chat] {response.chat}")
        else:
            print("Q: (silent)")
        print()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
