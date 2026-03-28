import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from integrations.github import search_github
from integrations.notion import search_notion
from integrations.slack import search_slack
from integrations.asana import search_asana

# ── Test cases ────────────────────────────────────────────────────────────────

POSITIVE_TESTS = [
    ("github", "Test pull request"),
    ("notion", "Test Quorum"),
    ("slack",  "Hi Vidyut"),
    ("slack",  "Testing this"),
    ("asana",  "dumber and dumber"),
]

NEGATIVE_TESTS = [
    ("github", "xkqz99_nonexistent_pr_abc"),
    ("notion", "xkqz99 nonexistent page abc"),
    ("slack",  "xkqz99_nonexistent_message_abc"),
    ("asana",  "xkqz99_nonexistent_task_abc"),
]

SOURCE_FN = {
    "github": search_github,
    "notion": search_notion,
    "slack":  search_slack,
    "asana":  search_asana,
}

# ── Runner ────────────────────────────────────────────────────────────────────

async def run_tests(tests, expect_results: bool):
    label = "POSITIVE" if expect_results else "NEGATIVE"
    print(f"\n{'='*60}")
    print(f"  {label} TESTS (expect {'results' if expect_results else 'no results'})")
    print(f"{'='*60}\n")

    passed = 0
    for source, query in tests:
        fn = SOURCE_FN[source]
        results = await fn(query, max_results=3)
        got_results = len(results) > 0

        if expect_results:
            ok = got_results
        else:
            ok = not got_results

        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {source} — '{query}'")
        if got_results:
            for r in results:
                print(f"         → {r.title}")
        else:
            print(f"         → no results")
        print()

        if ok:
            passed += 1

    print(f"  {passed}/{len(tests)} passed\n")
    return passed, len(tests)


async def main():
    p1, t1 = await run_tests(POSITIVE_TESTS, expect_results=True)
    p2, t2 = await run_tests(NEGATIVE_TESTS, expect_results=False)

    total_pass = p1 + p2
    total = t1 + t2
    print(f"{'='*60}")
    print(f"  TOTAL: {total_pass}/{total} passed")
    print(f"{'='*60}\n")


asyncio.run(main())
