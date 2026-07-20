# Recall NDJSON Streaming — Progress Ledger

**Branch:** `feature/recall-ndjson-streaming` · **Issue:** pajoma/cognee#2
**Spec:** `docs/superpowers/specs/2026-07-20-recall-ndjson-streaming-design.md` (v4 APPROVED)
**Plan:** `docs/superpowers/plans/2026-07-20-recall-ndjson-streaming-plan.md`
**Base:** `fbaba22af` (design docs only). Plan+ledger rebuilt 2026-07-20 (prior copy lost, never pushed).

**Env:** `.venv` uv CPython 3.12.13; core `-e .` + `pytest pytest-asyncio ruff` (docs extra skipped). Tests: `.venv/bin/python -m pytest`.

| # | Task | Status | Commit |
|---|------|--------|--------|
| 1 | ProgressEmitter module + unit tests | done | (this commit) |
| 2 | Streaming orchestrator stream.py | pending | — |
| 3 | Terminal error mapper + status_code | pending | — |
| 4 | Content negotiation router | pending | — |
| 5 | Emit points recall.py | pending | — |
| 6 | Emit points get_retriever_output.py | pending | — |
| 7 | Config keepalive interval | pending | — |
| 8 | MCP client NDJSON consumption | pending | — |
| 9 | Integration test e2e | pending | — |

## Log
- 2026-07-20: rebuilt plan + ledger from committed spec on Linux box (uv 3.12 venv). Starting Task 1.
