"""Unit tests for the recall progress emitter (Task 1).

Covers the design's emitter contract: no-op when unscoped, ordered/awaited enqueue with no
``seq``, contextvar propagation into ``asyncio.create_task``, sentinel identity, ``detail``
omission, scope reset, and bounded-queue backpressure.
"""

import asyncio

import pytest

from cognee.modules.recall.progress import (
    DEFAULT_QUEUE_MAXSIZE,
    SENTINEL,
    ProgressEmitter,
    emit,
    get_progress_emitter,
    progress_scope,
)


@pytest.mark.asyncio
async def test_emit_is_noop_without_scope():
    # No emitter bound: module-level emit returns immediately and raises nothing.
    assert get_progress_emitter() is None
    await emit("routing", "completed", {"search_type": "GRAPH_COMPLETION"})
    assert get_progress_emitter() is None


@pytest.mark.asyncio
async def test_scope_binds_and_resets():
    assert get_progress_emitter() is None
    async with progress_scope() as emitter:
        assert isinstance(emitter, ProgressEmitter)
        assert get_progress_emitter() is emitter
    # Contextvar reset on exit -> back to no-op path.
    assert get_progress_emitter() is None


@pytest.mark.asyncio
async def test_events_enqueue_in_order_without_seq():
    async with progress_scope() as emitter:
        await emit("session", "completed", {"count": 1})
        await emit("routing", "started")
        await emit("routing", "completed", {"search_type": "GRAPH_COMPLETION"})

    first = emitter.queue.get_nowait()
    second = emitter.queue.get_nowait()
    third = emitter.queue.get_nowait()

    assert first == {
        "type": "progress",
        "stage": "session",
        "status": "completed",
        "detail": {"count": 1},
    }
    # detail omitted when None.
    assert second == {"type": "progress", "stage": "routing", "status": "started"}
    assert "detail" not in second
    assert third["stage"] == "routing" and third["status"] == "completed"
    # seq is never stamped by the emitter (drain loop is the sole sequencer).
    assert all("seq" not in ev for ev in (first, second, third))
    assert emitter.queue.empty()


@pytest.mark.asyncio
async def test_contextvar_propagates_into_created_task():
    async with progress_scope() as emitter:
        # asyncio tasks copy the current context, so deep per-dataset tasks see the emitter.
        async def child():
            await emit("graph_retrieval", "completed", {"object_count": 3})

        await asyncio.create_task(child())

    event = emitter.queue.get_nowait()
    assert event["stage"] == "graph_retrieval"
    assert event["detail"] == {"object_count": 3}


@pytest.mark.asyncio
async def test_sentinel_is_distinct_identity():
    # Sentinel is an opaque marker, distinguishable from any progress event by identity.
    async with progress_scope() as emitter:
        await emit("session", "completed")
        await emitter.queue.put(SENTINEL)

    event = emitter.queue.get_nowait()
    marker = emitter.queue.get_nowait()
    assert event is not SENTINEL
    assert marker is SENTINEL


@pytest.mark.asyncio
async def test_bounded_queue_applies_backpressure():
    emitter = ProgressEmitter(maxsize=1)
    await emitter.emit("session", "started")  # fills the queue

    # Second put must block until capacity frees — proves producers suspend, never drop.
    blocked = asyncio.create_task(emitter.emit("session", "completed"))
    await asyncio.sleep(0)
    assert not blocked.done()

    emitter.queue.get_nowait()  # free one slot
    await asyncio.wait_for(blocked, timeout=1)
    assert blocked.done()


def test_default_queue_maxsize_is_bounded():
    assert isinstance(DEFAULT_QUEUE_MAXSIZE, int) and DEFAULT_QUEUE_MAXSIZE > 0
