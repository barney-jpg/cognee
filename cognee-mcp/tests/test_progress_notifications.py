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


def test_recall_tool_emits_progress_over_real_session(monkeypatch):
    # End-to-end through a real in-memory MCP ClientSession: call the registered `recall` tool
    # with a progress_callback (which sets progressToken) and assert notifications/progress are
    # delivered over the transport — exercising SDK context/token propagation and serialization,
    # not just the monkeypatched helper.
    from mcp.shared.memory import create_connected_server_and_client_session

    class _FakeClient:
        use_api = False

        async def recall(self, **kwargs):
            await asyncio.sleep(0.05)
            return []

    monkeypatch.setattr(server, "cognee_client", _FakeClient())
    monkeypatch.setenv("COGNEE_MCP_PROGRESS_INTERVAL", "0.01")

    progress_events = []

    async def _on_progress(progress, total, message):
        progress_events.append((progress, message))

    async def _run():
        # The in-memory harness drives the low-level protocol server: it calls
        # create_initialization_options(), which only the low-level object exposes.
        # `server.mcp` used to be the mcp SDK's FastMCP and could be passed directly;
        # since the move to the standalone `fastmcp` package it no longer forwards that
        # API, so reach through to the server it wraps. `_mcp_server` is private because
        # fastmcp offers no public accessor for it.
        async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
            return await client.call_tool("recall", {"query": "q"}, progress_callback=_on_progress)

    result = asyncio.run(_run())
    assert result.isError is False
    assert len(progress_events) >= 1
    assert progress_events[0][1] == "Recalling memory…"
