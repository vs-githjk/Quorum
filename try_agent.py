import asyncio, os
from dotenv import load_dotenv
load_dotenv()

from agent.context import MeetingContext
from agent.q_agent import QAgent
from integrations.asana import search_asana, create_asana_task
from integrations.slack import search_slack
from integrations.notion import search_notion
from integrations.github import search_github

def fmt(results):
    if not results:
        return "No results found."
    return "\n".join(f"- [{r.source}] {r.title}: {r.summary} ({r.url})" for r in results)

async def main():
    ctx = MeetingContext(history_file="/dev/null")
    agent = QAgent(context=ctx)

    async def tool_search_slack(meeting_id, query): return fmt(await search_slack(query))
    async def tool_search_notion(meeting_id, query): return fmt(await search_notion(query))
    async def tool_search_github(meeting_id, query): return fmt(await search_github(query))
    async def tool_search_asana(meeting_id, query): return fmt(await search_asana(query))
    async def tool_create_asana_task(meeting_id, title, notes=""): return await create_asana_task(title, notes)
    async def tool_log_decision(meeting_id, decision):
        ctx.add_decision(decision, meeting_id); return f"Logged: {decision}"
    async def tool_search_past_meetings(meeting_id, query): return ctx.search_past_meetings(query)

    agent.register_tools({
        "search_slack": tool_search_slack,
        "search_notion": tool_search_notion,
        "search_github": tool_search_github,
        "search_asana": tool_search_asana,
        "create_asana_task": tool_create_asana_task,
        "log_decision": tool_log_decision,
        "search_past_meetings": tool_search_past_meetings,
    })

    mid = "test-meeting"
    print("Type a message to Q (Ctrl+C to quit)\n")
    while True:
        text = input("> ").strip()
        if not text:
            continue
        response = await agent.run(text, mid, "You")
        print(f"Q: {response or '(silent)'}\n")

asyncio.run(main())
