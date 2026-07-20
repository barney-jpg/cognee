# Recall NDJSON Streaming — Design

**Date:** 2026-07-20
**Tracking issue:** pajoma/cognee#2 (companion: pajoma/cognee#1 — MCP progress relay)
**Branch:** `feature/recall-ndjson-streaming`

## Problem

`POST /api/v1/recall` returns a single blocking JSON body: the caller receives nothing
until the whole pipeline (session/graph retrieval + LLM generation) finishes. Long recalls
can run many seconds with no data on the wire, which trips downstream MCP/client/proxy idle
timeouts (the originating symptom: LibreChat's MCP request timeout aborting a recall that
was still running server-side). There is also no progress signal for UX or observability.

## Goal

Stream recall as **NDJSON** (newline-delimited JSON), selected via **content negotiation**
on the existing endpoint (no new route), delivering two things:

1. **Liveness (safety net):** a time-based keepalive so the stream never idles longer than a
   configured interval, guaranteeing recall cannot trip an idle timeout — regardless of how
   long any single stage (e.g. `llm_generation`) runs silently.
2. **Semantic stage events (richness):** per-stage `progress` events covering the full
   pipeline, including the `graph_retrieval` vs `llm_generation` split.

The synchronous JSON response path must remain **byte-for-byte unchanged** when no streaming
`Accept` header is present.

## Scope

- **In scope:** the cognee FastAPI streaming side (content negotiation, event production,
  keepalive, error framing) AND the `cognee-mcp` `CogneeClient.recall` consuming the NDJSON
  stream, collapsing it to the final result (progress/keepalive events ignored for now).
- **Out of scope (issue #1):** relaying the stage events to LibreChat as MCP
  `notifications/progress`. Issue #2 is the data source and the client plumbing; #1 turns the
  ignored events into user-visible progress. The user-facing LibreChat liveness benefit lands
  with #1.
- **Non-goals:** no new route; no change to search *types* or routing *decisions* (only
  reporting them); the existing JSON recall response is not removed or altered.

## Approach (selected: A — contextvar-bound emitter + queue bridge)

The graph sub-stages fire inside `get_retriever_output`, four calls below `recall()`, and
`authorized_search` is awaited as an opaque unit. A `ProgressEmitter` bound to a
`contextvar` lets deep code report without threading a callback through every signature; an
`asyncio.Queue` bridges the callback-style `emit()` to the top-level async generator that
produces NDJSON lines. When no emitter is in context (the normal JSON path), `emit()` is a
cheap no-op, so the synchronous path is behaviorally identical.

Rejected alternatives:
- **B — explicit `on_stage` callback threaded through the chain:** touches 4 signatures and
  every caller; more churn and positional-arg risk for the same result.
- **C — full async-generator refactor of the search chain:** deeply invasive to a hot path
  shared with `/search`; disproportionate regression risk.

## Architecture & components

### New module: `cognee/modules/recall/progress.py`

- `ProgressEmitter` — wraps an `asyncio.Queue`. `emit(stage, status, detail=None)` builds an
  event dict (adds a monotonic `seq`) and does `queue.put_nowait(...)`. Enqueues a sentinel
  on completion.
- `_progress_emitter: ContextVar[ProgressEmitter | None]` with `get_progress_emitter()` and a
  `progress_scope()` context manager that sets/resets it.
- Module-level `emit(stage, status, detail=None)` — resolves the contextvar; **no-op when
  unset**. This is what deep code calls, so callers never handle the emitter object directly.

### Streaming orchestrator: `cognee/api/v1/recall/stream.py` (helper `stream_recall_ndjson(...)`)

- Enters `progress_scope()`, launches `cognee_recall(...)` as an `asyncio.Task`.
- Drain loop races: queue items, a periodic keepalive tick, and task completion.
- Yields each event as `json.dumps(event) + "\n"`. On task completion: drains remaining
  events, yields exactly one terminal `result` or `error`, returns.

### Touched, minimally

- `cognee/api/v1/recall/routers/get_recall_router.py` — content-negotiation branch.
- `cognee/api/v1/recall/recall.py` — `emit(...)` at the source-loop stage boundaries
  (`session`, `trace`, `session_context`, `routing`, `normalization`).
- `cognee/modules/search/methods/get_retriever_output.py` — `emit(...)` at the existing three
  `new_span` phases → `graph_retrieval` (get_objects + get_context) and `llm_generation`
  (get_completion).
- `cognee-mcp/src/cognee_client.py` — `recall()` sends `Accept: application/x-ndjson`,
  consumes the stream, collapses to the final result.

### Data flow

```
route (Accept: application/x-ndjson)
  └─ progress_scope()  ── sets contextvar ─┐
  └─ task: cognee_recall(...)              │  emit() is a no-op if contextvar unset
        recall(): emit(session/trace/session_context/routing/normalization)
        get_retriever_output(): emit(graph_retrieval/llm_generation)
                                           ├──▶ asyncio.Queue ──▶ drain loop ──▶ NDJSON lines
        (+ keepalive timer) ───────────────┘
  └─ terminal {"type":"result"|"error"}
```

The synchronous JSON path never enters `progress_scope()`, so `emit()` is always a no-op
there.

## Event schema

Each line is one JSON object with a `type`:

```jsonc
// progress — one per stage boundary
{"type":"progress","stage":"routing","status":"completed","seq":4,
 "detail":{"search_type":"GRAPH_COMPLETION","confidence":3.0}}
// keepalive — time-based liveness, no semantics
{"type":"keepalive","seq":7,"elapsed_ms":20000}
// terminal — exactly one, always last
{"type":"result","data":[ /* list[RecallResponse], identical to the JSON body */ ]}
// ...or on failure:
{"type":"error","stage":"llm_generation","status_code":409,
 "message":"An error occurred during recall."}
```

- `status` ∈ `started` | `completed`. Skipped stages emit neither.
- `seq` — monotonic counter so a consumer can detect drops/ordering.
- `detail` — small, stage-specific, optional. Never large payloads (no context dumps, no
  vectors).

## Stage → emission point mapping

| Stage | Emitted from | detail |
|-------|--------------|--------|
| `session` | `recall._run_session` | completed: `{count}` |
| `trace` | `recall._run_trace` | completed: `{count}` |
| `session_context` | `recall._run_session_context` | completed: `{count}` |
| `routing` | `recall._run_graph` after `route_query()` (recall.py L540-549) | completed: `{search_type, confidence}`; `overridden: true` when an explicit `query_type` bypasses routing |
| `graph_retrieval` | `get_retriever_output` around `get_retrieved_objects` + `get_context_from_objects` (L65-89) | completed: `{object_count}` |
| `llm_generation` | `get_retriever_output` around `get_completion_from_context` (L94-106); **skipped when `only_context=True`** | `started` only |
| `normalization` | `recall._run_graph` around `normalize_search_payload` tagging (L592-598) | completed: `{result_count}` |

**Ordering:** with `ENABLE_BACKEND_ACCESS_CONTROL=false` (the target deployment) the
non-access-control branch runs a single retriever pass, so `graph_retrieval`/`llm_generation`
fire once. Under access control, per-dataset searches run concurrently via `asyncio` tasks,
so those two stages may repeat/interleave; events then carry a `dataset` field in `detail` to
disambiguate. The contextvar propagates into those tasks automatically (tasks copy the
current context).

## Content negotiation

- Add `request: Request` to the `recall` handler (get_recall_router.py L128-130). Parse
  `Accept`; a value containing `application/x-ndjson` selects streaming (case/`q`-value
  tolerant; `*/*` and absent → JSON).
- Streaming branch: `StreamingResponse(gen(), media_type="application/x-ndjson")` with
  `X-Accel-Buffering: no` and `Cache-Control: no-cache` so intermediaries don't buffer.
- Default branch: unchanged (`jsonable_encoder(results)`), same DTO, same auth dependency.

## Error handling

A `StreamingResponse` has already sent `200 OK`, so status cannot change mid-stream.

- **Before streaming starts:** auth (`Depends`) and DTO validation run before the handler
  body, so those keep normal HTTP status codes.
- **After streaming starts:** the orchestrator wraps the recall task in try/except and maps
  the same exception classes the JSON branch handles into a terminal `error` event carrying
  `status_code`:
  - `LLMPaymentRequiredError` → 402
  - `DatabaseNotCreatedError` / `UserNotFoundError` / `CogneeValidationError` → 422
  - `PermissionDeniedError` → terminal `result` with empty `data` (mirrors the JSON branch's
    empty-list behavior)
  - generic `Exception` → 409
- Exactly one terminal event (`result` or `error`) always closes the stream — no half-open
  streams, no exception escaping the generator.
- The queue is bounded; `put_nowait` on a full queue drops the **oldest keepalive** (never a
  `progress` or terminal event) to bound memory without losing semantics.

## MCP client consumption

`CogneeClient.recall` (cognee_client.py L532-552):

- Sends `Accept: application/x-ndjson`; uses `client.stream("POST", ...)` and iterates
  `response.aiter_lines()`, parsing each JSON line.
- For issue #2: **ignores** `progress`/`keepalive` events; returns the `result` event's
  `data`; raises on a terminal `error` event carrying the `status_code`. (Issue #1 swaps
  "ignore" for "relay as MCP progress notifications".)
- **Back-compat:** if the response `Content-Type` is `application/json` (older backend that
  ignored `Accept`), fall back to the existing `response.json()` path. A new MCP against an
  old API still works.

## Configuration

- `COGNEE_RECALL_KEEPALIVE_INTERVAL` — keepalive period in seconds (default `10`). `0`
  disables keepalive (stage events only).
- Bounded queue size — internal constant; keepalives are the only droppable event type.

## Testing

- **Unit (API):**
  - JSON branch unchanged when no `Accept` header (response equals current behavior).
  - NDJSON branch emits ordered `progress` stages + exactly one terminal `result`.
  - Injected failure yields a terminal `error` with the correct `status_code`; no exception
    escapes the generator.
  - A slow stage (mocked) produces ≥1 `keepalive`.
- **Unit (emitter):** `emit()` is a no-op with no scope; events enqueue in order within a
  scope; contextvar propagates into `asyncio.create_task`.
- **Unit (MCP client):** fed a canned NDJSON stream, `recall()` returns the `result` data and
  ignores progress/keepalive; JSON fallback path still works.
- **Integration:** end-to-end recall over NDJSON returns a result payload equal to the JSON
  path's for the same query.

## Risks & mitigations

- **Contextvar not propagating into per-dataset tasks** → `asyncio.create_task` copies the
  current context by default; covered by an emitter unit test.
- **Buffering proxies defeating liveness** → `X-Accel-Buffering: no` + `Cache-Control:
  no-cache`; NDJSON flushed per line.
- **Behavioral drift on the JSON path** → `emit()` no-op when unscoped; explicit test that the
  JSON branch output is unchanged.
- **Error-after-200 masking real status** → terminal `error` event carries `status_code`; MCP
  client maps it back to an exception.
