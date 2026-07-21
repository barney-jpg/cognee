"""Unit tests for the NDJSON streaming orchestrator (Tasks 2 + 3).

Covers the design's stream invariants: ordered progress + exactly one terminal, keepalive
enabled/disabled, seq monotonicity across all line kinds, terminal error mapping with
status_code preservation, permission-denied as an empty result, backpressure (no lost event),
and disconnect/saturation cleanup with no deadlock and no escaping exception.
"""

import asyncio
import json

import pytest

from cognee.api.v1.recall.stream import stream_recall_ndjson
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError
from cognee.modules.recall.progress import emit, get_progress_emitter
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError


async def _collect(gen):
    """Drain an async NDJSON generator into a list of parsed events."""
    events = []
    async for line in gen:
        assert line.endswith("\n")
        events.append(json.loads(line))
    return events


def _make_recall(events_to_emit=None, result=None, exc=None):
    """Build a fake recall coroutine that emits stage events then returns/raises."""

    async def _recall(**kwargs):
        for stage, status, detail in events_to_emit or []:
            await emit(stage, status, detail)
        if exc is not None:
            raise exc
        return result if result is not None else []

    return _recall


@pytest.mark.asyncio
async def test_progress_then_single_terminal_result():
    recall_fn = _make_recall(
        events_to_emit=[
            ("routing", "completed", {"search_type": "GRAPH_COMPLETION"}),
            ("graph_retrieval", "completed", {"object_count": 2}),
        ],
        result=[{"answer": "hi"}],
    )
    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=recall_fn)
    )

    kinds = [e["type"] for e in events]
    assert kinds == ["progress", "progress", "result"]
    assert events[-1]["data"] == [{"answer": "hi"}]
    # Exactly one terminal, always last.
    assert kinds.count("result") + kinds.count("error") == 1
    # Scope reset after the stream ends.
    assert get_progress_emitter() is None


@pytest.mark.asyncio
async def test_seq_is_monotonic_and_gap_free_across_all_lines():
    # Slow single stage so keepalives interleave with progress + terminal.
    async def slow_recall(**kwargs):
        await emit("routing", "started")
        await asyncio.sleep(0.05)
        await emit("routing", "completed", {"search_type": "X"})
        return []

    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0.01, recall_fn=slow_recall)
    )
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(1, len(seqs) + 1))  # strictly increasing, gap-free, from 1
    assert any(e["type"] == "keepalive" for e in events)
    assert events[-1]["type"] == "result"


@pytest.mark.asyncio
async def test_keepalive_disabled_emits_zero_and_does_not_spin():
    async def slow_recall(**kwargs):
        await asyncio.sleep(0.05)
        return []

    events = await asyncio.wait_for(
        _collect(
            stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=slow_recall)
        ),
        timeout=2,
    )
    assert not any(e["type"] == "keepalive" for e in events)
    assert [e["type"] for e in events] == ["result"]


@pytest.mark.asyncio
async def test_terminal_error_generic_is_409_and_no_exception_escapes():
    recall_fn = _make_recall(exc=RuntimeError("boom"))
    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=recall_fn)
    )
    assert events[-1] == {
        "type": "error",
        "status_code": 409,
        "message": "An error occurred during recall.",
        "seq": events[-1]["seq"],
    }
    assert get_progress_emitter() is None


@pytest.mark.asyncio
async def test_payment_required_maps_to_402():
    recall_fn = _make_recall(exc=LLMPaymentRequiredError())
    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=recall_fn)
    )
    assert events[-1]["type"] == "error"
    assert events[-1]["status_code"] == 402
    assert events[-1]["message"] == "Token budget exhausted"


@pytest.mark.asyncio
async def test_status_code_preserved_for_validation_subclass_404():
    # DatasetNotFoundError is a CogneeValidationError carrying 404 — must not collapse to 422.
    recall_fn = _make_recall(exc=DatasetNotFoundError("missing"))
    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=recall_fn)
    )
    assert events[-1]["type"] == "error"
    assert events[-1]["status_code"] == 404


@pytest.mark.asyncio
async def test_generic_validation_error_defaults_422():
    recall_fn = _make_recall(exc=CogneeValidationError("bad", status_code=422))
    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=recall_fn)
    )
    assert events[-1]["status_code"] == 422


@pytest.mark.asyncio
async def test_permission_denied_is_empty_result_not_error():
    recall_fn = _make_recall(exc=PermissionDeniedError())
    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=recall_fn)
    )
    assert events[-1]["type"] == "result"
    assert events[-1]["data"] == []


@pytest.mark.asyncio
async def test_error_stage_attribution_when_attached():
    exc = RuntimeError("boom")
    exc.stage = "llm_generation"
    recall_fn = _make_recall(exc=exc)
    events = await _collect(
        stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=recall_fn)
    )
    assert events[-1]["stage"] == "llm_generation"


@pytest.mark.asyncio
async def test_backpressure_no_progress_event_lost():
    # Emit far more events than the queue bound; awaited put means none are dropped.
    n = 20

    async def many(**kwargs):
        for i in range(n):
            await emit("graph_retrieval", "completed", {"i": i})
        return []

    events = await _collect(
        stream_recall_ndjson(
            recall_kwargs={}, keepalive_interval=0, queue_maxsize=2, recall_fn=many
        )
    )
    progress = [e for e in events if e["type"] == "progress"]
    assert [e["detail"]["i"] for e in progress] == list(range(n))  # ordered, none lost


@pytest.mark.asyncio
async def test_disconnect_cancels_task_and_resets_scope():
    started = asyncio.Event()
    cancelled = {"value": False}

    async def hang(**kwargs):
        await emit("routing", "started")
        started.set()
        try:
            await asyncio.sleep(100)  # simulate a long stage
        except asyncio.CancelledError:
            cancelled["value"] = True
            raise
        return []

    gen = stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=hang)
    first = await gen.__anext__()  # one progress line
    assert json.loads(first)["type"] == "progress"
    await started.wait()

    # Simulate client disconnect: close the generator; cleanup must terminate in bounded time.
    await asyncio.wait_for(gen.aclose(), timeout=2)
    assert cancelled["value"] is True
    assert get_progress_emitter() is None


@pytest.mark.asyncio
async def test_saturation_plus_disconnect_no_deadlock():
    # Fill the bounded queue, then disconnect. Cleanup drains so the blocked producer unblocks.
    producing = asyncio.Event()

    async def flood(**kwargs):
        producing.set()
        for i in range(100):  # far exceeds maxsize -> producer blocks on put
            await emit("graph_retrieval", "completed", {"i": i})
        return []

    gen = stream_recall_ndjson(
        recall_kwargs={}, keepalive_interval=0, queue_maxsize=1, recall_fn=flood
    )
    first = await gen.__anext__()
    assert json.loads(first)["type"] == "progress"
    await producing.wait()
    await asyncio.sleep(0.01)  # let the producer saturate the queue

    await asyncio.wait_for(gen.aclose(), timeout=2)  # must not hang
    assert get_progress_emitter() is None


@pytest.mark.asyncio
async def test_body_raising_cancellederror_still_closes_with_one_terminal():
    # A bare CancelledError from the recall body (not our cleanup cancel) must not escape the
    # generator; the stream still closes with exactly one terminal event.
    async def canceller(**kwargs):
        await emit("routing", "started")
        raise asyncio.CancelledError()

    events = await asyncio.wait_for(
        _collect(stream_recall_ndjson(recall_kwargs={}, keepalive_interval=0, recall_fn=canceller)),
        timeout=2,
    )
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "error"
    assert kinds.count("result") + kinds.count("error") == 1
    assert events[-1]["status_code"] == 409
    assert get_progress_emitter() is None
