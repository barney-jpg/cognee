"""Unit tests for keepalive-interval configuration (Task 7)."""

import asyncio
import json

import pytest

from cognee.api.v1.recall.stream import (
    DEFAULT_KEEPALIVE_INTERVAL,
    KEEPALIVE_INTERVAL_ENV,
    resolve_keepalive_interval,
    stream_recall_ndjson,
)
from cognee.modules.recall.progress import emit


def test_unset_returns_default(monkeypatch):
    monkeypatch.delenv(KEEPALIVE_INTERVAL_ENV, raising=False)
    assert resolve_keepalive_interval() == DEFAULT_KEEPALIVE_INTERVAL


def test_blank_returns_default(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_INTERVAL_ENV, "   ")
    assert resolve_keepalive_interval() == DEFAULT_KEEPALIVE_INTERVAL


def test_numeric_value_parsed(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_INTERVAL_ENV, "5")
    assert resolve_keepalive_interval() == 5.0


def test_zero_disables(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_INTERVAL_ENV, "0")
    assert resolve_keepalive_interval() == 0.0


def test_negative_honored(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_INTERVAL_ENV, "-1")
    assert resolve_keepalive_interval() == -1.0


def test_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_INTERVAL_ENV, "not-a-number")
    assert resolve_keepalive_interval() == DEFAULT_KEEPALIVE_INTERVAL


@pytest.mark.asyncio
async def test_orchestrator_reads_env_when_interval_omitted(monkeypatch):
    # Env disables keepalives; a slow stage must emit zero keepalive lines when the caller
    # passes no explicit interval (so the orchestrator resolves from the environment).
    monkeypatch.setenv(KEEPALIVE_INTERVAL_ENV, "0")

    async def slow_recall(**kwargs):
        await asyncio.sleep(0.05)
        await emit("routing", "completed", {"search_type": "X"})
        return []

    events = []
    async for line in stream_recall_ndjson(recall_kwargs={}, recall_fn=slow_recall):
        events.append(json.loads(line))

    assert not any(e["type"] == "keepalive" for e in events)
    assert events[-1]["type"] == "result"
