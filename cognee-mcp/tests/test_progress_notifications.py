"""Tests for MCP progress notifications during recall/search (issue #1)."""

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

server = importlib.import_module("src.server")


class _FakeSession:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def send_progress_notification(
        self, progress_token, progress, total=None, message=None, related_request_id=None
    ):
        if self.fail:
            raise RuntimeError("notify boom")
        self.calls.append({"token": progress_token, "progress": progress, "message": message})


def test_interval_config_default_and_override(monkeypatch):
    monkeypatch.delenv("COGNEE_MCP_PROGRESS_INTERVAL", raising=False)
    assert server._progress_interval() == server.DEFAULT_PROGRESS_INTERVAL
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0.01")
    assert server._progress_interval() == 0.01
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0")
    assert server._progress_interval() == 0.0
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "bogus")
    assert server._progress_interval() == server.DEFAULT_PROGRESS_INTERVAL


def test_emits_at_least_one_progress_before_result(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(server, "_progress_target", lambda: (session, "tok"))
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0.01")

    async def slow():
        await asyncio.sleep(0.05)
        return "done"

    result = asyncio.run(server._with_progress(slow(), label="Working"))
    assert result == "done"
    assert len(session.calls) >= 1
    # Progress increments and carries a human-readable message.
    assert session.calls[0]["message"] == "Working…"
    assert session.calls[0]["progress"] == 1


def test_no_progress_token_means_no_notifications(monkeypatch):
    monkeypatch.setattr(server, "_progress_target", lambda: (None, None))
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0.01")

    async def slow():
        await asyncio.sleep(0.03)
        return "unchanged"

    # Result is returned unchanged; no session means nothing could be notified.
    assert asyncio.run(server._with_progress(slow(), label="Working")) == "unchanged"


def test_disabled_interval_emits_nothing(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(server, "_progress_target", lambda: (session, "tok"))
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0")

    async def slow():
        await asyncio.sleep(0.03)
        return "done"

    assert asyncio.run(server._with_progress(slow(), label="Working")) == "done"
    assert session.calls == []


def test_notification_failure_does_not_break_result(monkeypatch):
    session = _FakeSession(fail=True)
    monkeypatch.setattr(server, "_progress_target", lambda: (session, "tok"))
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0.01")

    async def slow():
        await asyncio.sleep(0.05)
        return "resilient"

    # send_progress_notification raises every tick, yet the result is still returned.
    assert asyncio.run(server._with_progress(slow(), label="Working")) == "resilient"


def test_recall_tool_emits_progress_for_slow_call(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(server, "_progress_target", lambda: (session, "tok"))
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0.01")

    class _FakeClient:
        use_api = False

        async def recall(self, **kwargs):
            await asyncio.sleep(0.05)
            return []

    monkeypatch.setattr(server, "cognee_client", _FakeClient())

    result = asyncio.run(server.recall(query="q"))
    assert isinstance(result, list) and result and result[0].type == "text"
    assert len(session.calls) >= 1
