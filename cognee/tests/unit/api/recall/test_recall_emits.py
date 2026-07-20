"""Emit-point tests for recall() stage boundaries (Task 5).

Drives recall() with its heavy dependencies mocked, inside a progress scope, and asserts the
routing / normalization / session stage events fire with the documented detail.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.recall.progress import progress_scope
from cognee.modules.search.types import SearchType

recall_mod = importlib.import_module("cognee.api.v1.recall.recall")
serve_state_mod = importlib.import_module("cognee.api.v1.serve.state")
search_mod = importlib.import_module("cognee.modules.search.methods.search")
utils_mod = importlib.import_module("cognee.shared.utils")

MOCK_USER = SimpleNamespace(id=uuid4(), email="t@example.com", is_active=True, tenant_id=uuid4())


async def _drain(emitter):
    events = []
    while not emitter.queue.empty():
        events.append(emitter.queue.get_nowait())
    return events


@pytest.fixture(autouse=True)
def _mute_side_effects(monkeypatch):
    # Telemetry, remote-client lookup, and the session-user contextvar are irrelevant here.
    # Patch module objects (not dotted strings) to sidestep the package re-export shadowing.
    monkeypatch.setattr(utils_mod, "send_telemetry", lambda *a, **k: None)
    monkeypatch.setattr(serve_state_mod, "get_remote_client", lambda: None)
    monkeypatch.setattr(recall_mod, "set_session_user_context_variable", AsyncMock())


@pytest.mark.asyncio
async def test_graph_scope_emits_routing_and_normalization(monkeypatch):
    monkeypatch.setattr(search_mod, "authorized_search", AsyncMock(return_value=[]))
    async with progress_scope() as emitter:
        await recall_mod.recall(
            query_text="q",
            query_type=SearchType.GRAPH_COMPLETION,
            scope="graph",
            user=MOCK_USER,
        )
        events = await _drain(emitter)

    by_stage = {e["stage"]: e for e in events}
    assert "routing" in by_stage
    assert by_stage["routing"]["status"] == "completed"
    assert by_stage["routing"]["detail"]["search_type"] == "GRAPH_COMPLETION"
    # Explicit query_type -> routing marked overridden.
    assert by_stage["routing"]["detail"]["overridden"] is True

    assert by_stage["normalization"]["status"] == "completed"
    assert by_stage["normalization"]["detail"]["result_count"] == 0


@pytest.mark.asyncio
async def test_session_scope_emits_session_count(monkeypatch):
    monkeypatch.setattr(recall_mod, "_search_session", AsyncMock(return_value=["e1", "e2"]))
    async with progress_scope() as emitter:
        await recall_mod.recall(
            query_text="q",
            scope="session",
            session_id="s1",
            user=MOCK_USER,
        )
        events = await _drain(emitter)

    session = [e for e in events if e["stage"] == "session"]
    assert len(session) == 1
    assert session[0]["status"] == "completed"
    assert session[0]["detail"]["count"] == 2
