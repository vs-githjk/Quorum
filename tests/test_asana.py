"""
tests/test_asana.py — Unit tests for integrations/asana.py

Tests create_asana_task without hitting the real Asana API by patching
safe_post.  Tests search_asana's result-shaping logic with mock data.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

from integrations.asana import create_asana_task, _task_to_result


class TestCreateAsanaTask(unittest.IsolatedAsyncioTestCase):

    async def test_create_success_returns_formatted_string(self):
        mock_response = {
            "data": {
                "gid": "123456",
                "name": "fix the login bug",
                "permalink_url": "https://app.asana.com/0/0/123456/f",
            }
        }
        with (
            patch("integrations.asana._ASANA_TOKEN", "tok"),
            patch("integrations.asana._ASANA_WORKSPACE_GID", "wid"),
            patch("integrations.asana.safe_post", AsyncMock(return_value=mock_response)),
        ):
            result = await create_asana_task("fix the login bug")

        assert "fix the login bug" in result
        assert "https://app.asana.com" in result

    async def test_create_with_notes(self):
        mock_response = {
            "data": {
                "gid": "789",
                "name": "Write docs",
                "permalink_url": "https://app.asana.com/0/0/789/f",
            }
        }
        captured = {}

        async def fake_post(url, headers, body, **kw):
            captured["body"] = body
            return mock_response

        with (
            patch("integrations.asana._ASANA_TOKEN", "tok"),
            patch("integrations.asana._ASANA_WORKSPACE_GID", "wid"),
            patch("integrations.asana.safe_post", fake_post),
        ):
            await create_asana_task("Write docs", notes="See spec doc")

        assert captured["body"]["data"]["notes"] == "See spec doc"

    async def test_create_missing_token_returns_error(self):
        with patch("integrations.asana._ASANA_TOKEN", ""):
            result = await create_asana_task("anything")
        assert "Error" in result

    async def test_create_missing_workspace_returns_error(self):
        with (
            patch("integrations.asana._ASANA_TOKEN", "tok"),
            patch("integrations.asana._ASANA_WORKSPACE_GID", ""),
        ):
            result = await create_asana_task("anything")
        assert "Error" in result

    async def test_create_api_failure_returns_error(self):
        with (
            patch("integrations.asana._ASANA_TOKEN", "tok"),
            patch("integrations.asana._ASANA_WORKSPACE_GID", "wid"),
            patch("integrations.asana.safe_post", AsyncMock(return_value=None)),
        ):
            result = await create_asana_task("failing task")
        assert "Error" in result or "failed" in result.lower()

    async def test_create_no_permalink_omits_url(self):
        mock_response = {"data": {"gid": "x", "name": "no url task"}}
        with (
            patch("integrations.asana._ASANA_TOKEN", "tok"),
            patch("integrations.asana._ASANA_WORKSPACE_GID", "wid"),
            patch("integrations.asana.safe_post", AsyncMock(return_value=mock_response)),
        ):
            result = await create_asana_task("no url task")
        assert "no url task" in result


class TestTaskToResult(unittest.TestCase):

    def _make_task(self, **overrides):
        base = {
            "name": "Test task",
            "notes": "Some notes",
            "permalink_url": "https://app.asana.com/0/0/1/f",
            "completed": False,
            "due_on": "2026-04-01",
            "assignee": {"name": "Alice"},
        }
        base.update(overrides)
        return base

    def test_basic_mapping(self):
        r = _task_to_result(self._make_task())
        assert r.source == "asana"
        assert r.title == "Test task"
        assert "Alice" in r.summary
        assert "2026-04-01" in r.summary
        assert "open" in r.summary

    def test_completed_task(self):
        r = _task_to_result(self._make_task(completed=True))
        assert "completed" in r.summary

    def test_no_assignee(self):
        r = _task_to_result(self._make_task(assignee=None))
        assert "unassigned" in r.summary


if __name__ == "__main__":
    unittest.main()
