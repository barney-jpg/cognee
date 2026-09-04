"""Progress emission for streaming recall.

A ``ProgressEmitter`` bound to a :class:`contextvars.ContextVar` lets code deep in the recall
pipeline report per-stage progress without threading a callback through every signature. The
emitter wraps a **bounded** ``asyncio.Queue``; the streaming orchestrator drains that queue,
stamps a monotonic ``seq`` on each line, and produces NDJSON.

On the normal (synchronous JSON) path no emitter is bound, so the module-level :func:`emit` is a
cheap no-op and the pipeline behaves identically.

Design: ``docs/superpowers/specs/2026-07-20-recall-ndjson-streaming-design.md``.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Iterator, Optional

# Bounded-queue default. Because producers ``await queue.put(...)``, the bound governs
# backpressure sensitivity only, never correctness: no event is ever dropped. The streaming
# orchestrator (Task 7) may override this from configuration.
DEFAULT_QUEUE_MAXSIZE = 256

# Authoritative end-of-stream marker. The orchestrator's wrapper enqueues this from its
# ``finally`` *after* the recall body has produced a result or raised, so a dequeued SENTINEL
# means the task is already returning. Identity comparison only.
SENTINEL = object()


class ProgressEmitter:
    """Bridges async ``emit`` calls to the orchestrator via a bounded queue.

    ``emit`` builds the event dict **without** ``seq`` — the drain loop is the single output
    boundary and the sole sequencer, so one monotonic sequence spans progress, keepalive, and
    terminal events (keepalives are generated outside this queue and would collide with any
    per-producer counter).

    ``await queue.put(...)`` provides real backpressure: if the drain loop falls behind,
    producers suspend rather than dropping semantic events.
    """

    def __init__(self, maxsize: int = DEFAULT_QUEUE_MAXSIZE):
        self.queue: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=maxsize)

    async def emit(
        self,
        stage: str,
        status: str,
        detail: Optional[dict] = None,
    ) -> None:
        """Enqueue a ``progress`` event. ``detail`` is omitted when ``None``."""
        event: dict = {"type": "progress", "stage": stage, "status": status}
        if detail is not None:
            event["detail"] = detail
        await self.queue.put(event)


_progress_emitter: ContextVar[Optional[ProgressEmitter]] = ContextVar(
    "cognee_progress_emitter", default=None
)


def get_progress_emitter() -> Optional[ProgressEmitter]:
    """Return the emitter bound to the current context, or ``None`` on the synchronous path."""
    return _progress_emitter.get()


@asynccontextmanager
async def progress_scope(
    maxsize: int = DEFAULT_QUEUE_MAXSIZE,
) -> AsyncIterator[ProgressEmitter]:
    """Bind a fresh :class:`ProgressEmitter` for the duration of the block.

    Yields the emitter (the orchestrator drains ``emitter.queue`` and enqueues :data:`SENTINEL`
    onto it). Resets the contextvar on exit so the synchronous path stays a no-op afterward.
    """
    emitter = ProgressEmitter(maxsize=maxsize)
    token = _progress_emitter.set(emitter)
    try:
        yield emitter
    finally:
        _progress_emitter.reset(token)


async def emit(stage: str, status: str, detail: Optional[dict] = None) -> None:
    """Emit a progress event to the bound emitter, or no-op when none is bound.

    This is what deep pipeline code awaits, so callers never touch the emitter object directly.
    """
    emitter = _progress_emitter.get()
    if emitter is None:
        return
    await emitter.emit(stage, status, detail)


# The dataset a stage runs against. Bound once by the search entry point and read by the retrieval
# internals, for the same reason the emitter is a contextvar: the graph_retrieval / llm_generation
# boundaries sit inside upstream functions whose signatures this fork should not grow.
_progress_dataset: ContextVar[Optional[str]] = ContextVar("cognee_progress_dataset", default=None)


@contextmanager
def progress_dataset(name: Optional[str]) -> Iterator[None]:
    """Bind the dataset name that stage details and error attribution are tagged with."""
    token = _progress_dataset.set(name)
    try:
        yield
    finally:
        _progress_dataset.reset(token)


def stage_detail(**extra) -> dict:
    """Build a stage detail dict, tagged with the bound dataset name when there is one."""
    detail = dict(extra)
    dataset = _progress_dataset.get()
    if dataset:
        detail["dataset"] = dataset
    return detail


def attach_stage(exc: BaseException, stage: str) -> None:
    """Best-effort stage/dataset attribution on a failing exception for terminal ``error.stage``."""
    try:
        if getattr(exc, "stage", None) is None:
            exc.stage = stage
        dataset = _progress_dataset.get()
        if dataset and getattr(exc, "dataset", None) is None:
            exc.dataset = dataset
    except Exception:  # some builtins forbid attribute assignment — attribution is optional
        pass
