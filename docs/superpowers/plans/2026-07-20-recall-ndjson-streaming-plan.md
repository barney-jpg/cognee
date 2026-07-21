# Recall NDJSON Streaming — Implementation Plan

**Date:** 2026-07-20
**Spec:** `docs/superpowers/specs/2026-07-20-recall-ndjson-streaming-design.md` (v4, APPROVED)
**Tracking issue:** pajoma/cognee#2 (companion #1 — MCP progress relay)
**Branch:** `feature/recall-ndjson-streaming`

Plan reconstructed from the committed v4 design after the original plan/ledger were lost
(never pushed from a separate checkout). Base HEAD: `fbaba22af` (design docs only).

## Task ordering & dependencies

```
1 emitter ──▶ 2 orchestrator ──▶ 4 negotiation ──▶ 8 MCP client ──▶ 9 integration
                    │                  ▲
                    ├── 3 error mapper ┘
                    ├── 5 recall.py emits
                    ├── 6 retriever emits
                    └── 7 config
```

- **Task 1 — ProgressEmitter** (`cognee/modules/recall/progress.py`)
  Bounded `asyncio.Queue`; `async emit(stage,status,detail)` awaits `put` (backpressure, no
  `seq`); `_progress_emitter: ContextVar`; `get_progress_emitter()`; async `progress_scope()`;
  module-level no-op `emit()`. Unit tests: no-op unscoped, ordered enqueue, contextvar copies
  into `create_task`, sentinel constant.

- **Task 2 — Orchestrator** (`cognee/api/v1/recall/stream.py`)
  `stream_recall_ndjson(...)` async generator. Wrapper task enqueues authoritative SENTINEL in
  `finally`. Drain loop stamps monotonic `seq` on every line; keepalive via `wait_for(get,
  interval)` when `interval>0`, untimed `get()` when `interval<=0`. On sentinel: stop reading,
  `await recall_task`, yield one terminal. `finally`: cancel task + concurrently drain queue
  (deadlock guard), log+suppress non-Cancelled exceptions, reset scope.

- **Task 3 — Error mapper** (in `stream.py`)
  Mirror JSON branch exactly: `LLMPaymentRequiredError`→402; validation family →
  `getattr(e,"status_code",422)` (so `DatasetNotFoundError`→404); `PermissionDeniedError`→
  terminal `result` empty `data`; generic→409. Optional `error.stage` only when attached at
  failure site.

- **Task 4 — Content negotiation** (`cognee/api/v1/recall/routers/get_recall_router.py`)
  Add `request: Request`; parse `Accept` (media ranges + `q`); NDJSON iff `application/x-ndjson`
  effective `q>0`. `StreamingResponse(media_type="application/x-ndjson")` +
  `X-Accel-Buffering: no` + `Cache-Control: no-cache`. JSON branch byte-for-byte unchanged.

- **Task 5 — recall.py emits** — session, trace, session_context, routing
  (`{search_type,confidence}`, `overridden:true` on explicit `query_type`), normalization
  (`{result_count}`), coarse `remote` around remote-client early return.

- **Task 6 — get_retriever_output.py emits** — `graph_retrieval` (after `should_answer`,
  `{object_count}`, skip on early answer), `llm_generation` (`started` only, skip when
  `only_context=True` / early answer). Attach stage on failure for `error.stage`.

- **Task 7 — Config** — `COGNEE_RECALL_KEEPALIVE_INTERVAL` (default 10; `<=0` disables).
  Bounded queue size internal constant.

- **Task 8 — MCP client** (`cognee-mcp/src/cognee_client.py`) — `recall()` sends
  `Accept: application/x-ndjson`, streams `aiter_lines`, ignores progress/keepalive, returns
  `result.data`, raises typed exception carrying `status_code` on terminal `error`. JSON
  Content-Type back-compat fallback.

- **Task 9 — Integration** — end-to-end NDJSON result equals JSON path for same query;
  started/terminal invariants; `seq` monotonic & gap-free across full run.

## Per-task procedure

Each task: one implementer pass → run its tests (`.venv/bin/python -m pytest ...`) → task
review → commit with trailer `Co-Authored-By: Claude Opus 4.8 (1M context)`. Update
`.superpowers/sdd/progress.md` after each commit.

## Environment

`.venv` = uv-managed CPython 3.12.13. Installed: `-e .` core + `pytest pytest-asyncio ruff`
(docs extra skipped — `mkdocs-minify`/`csscompressor` fail to build). Run tests via
`.venv/bin/python -m pytest`.
