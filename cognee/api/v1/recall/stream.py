"""Streaming orchestrator for NDJSON recall (Tasks 2 + 3).

``stream_recall_ndjson`` runs the recall coroutine in a separate task and yields NDJSON lines:
per-stage ``progress`` events (from the contextvar-bound emitter), time-based ``keepalive``
lines, and exactly one terminal ``result``/``error``. The drain loop is the single output
boundary and the sole sequencer — it stamps a monotonic ``seq`` on every line.

Key invariants (see the v4 design doc):
- The wrapper enqueues the authoritative :data:`SENTINEL` from its ``finally`` *after* the recall
  body has produced a result or raised, so a dequeued sentinel means the task is already
  returning; the loop never re-reads the queue after seeing it.
- On generator close / client disconnect, the ``finally`` cancels the recall task and drains the
  queue *concurrently* while awaiting it, so a producer (or the sentinel ``put``) blocked on a
  full bounded queue cannot deadlock. Non-``CancelledError`` task exceptions are logged and
  suppressed — nothing is re-raised from the generator.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, Optional

from fastapi.encoders import jsonable_encoder

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.modules.recall.progress import (
    DEFAULT_QUEUE_MAXSIZE,
    SENTINEL,
    progress_scope,
)
from cognee.modules.users.exceptions.exceptions import (
    PermissionDeniedError,
    UserNotFoundError,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Keepalive period in seconds. ``<= 0`` disables keepalives (Task 7 overrides from config).
DEFAULT_KEEPALIVE_INTERVAL = 10.0

NDJSON_MEDIA_TYPE = "application/x-ndjson"


def _line(event: dict) -> str:
    """Serialize one event as a single NDJSON line."""
    return json.dumps(event, separators=(",", ":")) + "\n"


def map_exception_to_terminal(exc: BaseException) -> dict:
    """Map a recall exception to a terminal event, mirroring the JSON branch exactly.

    ``status_code`` is preserved (not hardcoded) so a validation subclass such as
    ``DatasetNotFoundError`` (404) reports the same code the JSON response would. An optional
    ``stage`` is included only when the failing layer attached it at the failure site.
    """
    if isinstance(exc, PermissionDeniedError):
        # Mirror the JSON branch's empty-list behavior — a terminal result, not an error.
        return {"type": "result", "data": []}

    if isinstance(exc, LLMPaymentRequiredError):
        event: dict = {"type": "error", "status_code": 402, "message": "Token budget exhausted"}
    elif isinstance(exc, (DatabaseNotCreatedError, UserNotFoundError, CogneeValidationError)):
        event = {
            "type": "error",
            "status_code": getattr(exc, "status_code", 422),
            "message": "Recall prerequisites not met",
        }
    else:
        event = {
            "type": "error",
            "status_code": 409,
            "message": "An error occurred during recall.",
        }

    stage = getattr(exc, "stage", None)
    if stage:
        event["stage"] = stage
    dataset = getattr(exc, "dataset", None)
    if dataset:
        event["dataset"] = dataset
    return event


async def _drain(queue: "asyncio.Queue[Any]") -> None:
    """Discard queue items until cancelled — frees blocked producers during cleanup."""
    while True:
        await queue.get()


async def stream_recall_ndjson(
    *,
    recall_kwargs: dict,
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL,
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    recall_fn: Optional[Callable[..., Any]] = None,
) -> AsyncIterator[str]:
    """Yield NDJSON lines for a recall.

    ``recall_fn`` defaults to the real ``cognee.recall`` (imported lazily to avoid an import
    cycle); tests inject a fake that emits stage events. ``recall_kwargs`` are forwarded to it
    verbatim.
    """
    if recall_fn is None:
        from cognee.api.v1.recall import recall as recall_fn  # lazy: breaks import cycle

    start = time.monotonic()
    seq = 0

    async with progress_scope(maxsize=queue_maxsize) as emitter:
        queue = emitter.queue
        outcome: dict = {}

        async def _wrapper() -> None:
            # CancelledError propagates (not captured) so cleanup can observe it; the sentinel
            # is enqueued in finally *after* the body settles, making it authoritative.
            try:
                outcome["result"] = await recall_fn(**recall_kwargs)
            except Exception as exc:  # noqa: BLE001 — captured for the terminal mapper
                outcome["error"] = exc
            finally:
                await queue.put(SENTINEL)

        recall_task = asyncio.create_task(_wrapper())

        try:
            while True:
                if keepalive_interval > 0:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
                    except asyncio.TimeoutError:
                        seq += 1
                        elapsed_ms = int((time.monotonic() - start) * 1000)
                        yield _line({"type": "keepalive", "seq": seq, "elapsed_ms": elapsed_ms})
                        continue
                else:
                    # Disabled: untimed get(), so no keepalives and no spin.
                    item = await queue.get()

                if item is SENTINEL:
                    # Authoritative: the body has settled; awaiting the task resolves at once.
                    # A bare CancelledError from the body itself (BaseException, so not captured
                    # by the wrapper) would otherwise escape here — swallow it and still close
                    # the stream with exactly one terminal, per the consumer invariant.
                    try:
                        await recall_task
                    except asyncio.CancelledError:
                        pass
                    seq += 1
                    if "error" in outcome:
                        terminal = map_exception_to_terminal(outcome["error"])
                    elif "result" in outcome:
                        terminal = {"type": "result", "data": jsonable_encoder(outcome["result"])}
                    else:
                        # Body cancelled before settling: no result and no captured error.
                        terminal = map_exception_to_terminal(RuntimeError("recall cancelled"))
                    terminal["seq"] = seq
                    yield _line(terminal)
                    return

                # Progress event: stamp seq at the single output boundary.
                seq += 1
                item["seq"] = seq
                yield _line(item)
        finally:
            # Client disconnect / GC: stop the recall task without orphaning DB/LLM work and
            # without deadlocking on a full queue (drain concurrently while awaiting).
            if not recall_task.done():
                recall_task.cancel()
                drainer = asyncio.create_task(_drain(queue))
                try:
                    await recall_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001 — observed, logged, suppressed
                    logger.error("Recall task failed during stream cleanup: %s", exc, exc_info=True)
                finally:
                    drainer.cancel()
                    try:
                        await drainer
                    except asyncio.CancelledError:
                        pass
