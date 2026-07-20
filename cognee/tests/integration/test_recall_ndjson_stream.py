"""End-to-end integration for NDJSON recall (Task 9).

Exercises the full server path — content negotiation, orchestrator, emitter, seq stamping,
terminal framing — via FastAPI's TestClient, mocking only the recall computation so the test
runs without a real graph/LLM. Asserts the NDJSON payload equals the JSON path's for the same
query and that the documented stream invariants hold.
"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cognee.api.v1.recall as recall_pkg
from cognee.api.v1.recall.routers.get_recall_router import get_recall_router
from cognee.api.v1.recall.stream import NDJSON_MEDIA_TYPE
from cognee.modules.recall.progress import emit
from cognee.modules.recall.types.RecallResponse import ResponseSessionContextEntry
from cognee.modules.users.methods import get_authenticated_user

MOCK_USER = SimpleNamespace(id=uuid4(), email="t@example.com", is_active=True, tenant_id=uuid4())


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_recall_router(), prefix="/recall")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


def _parse_ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_ndjson_payload_equals_json_payload(client):
    payload = [
        ResponseSessionContextEntry(
            source="session_context", content="answer", context_profile="qa"
        )
    ]

    async def fake_recall(**kwargs):
        await emit("routing", "completed", {"search_type": "GRAPH_COMPLETION"})
        await emit("graph_retrieval", "completed", {"object_count": 1})
        await emit("llm_generation", "started")
        return payload

    recall_pkg.recall = fake_recall

    json_resp = client.post("/recall", json={"query": "q"})
    ndjson_resp = client.post("/recall", json={"query": "q"}, headers={"Accept": NDJSON_MEDIA_TYPE})

    assert json_resp.status_code == 200
    events = _parse_ndjson(ndjson_resp.text)

    # Terminal result data is identical to the JSON body for the same query.
    assert events[-1]["type"] == "result"
    assert events[-1]["data"] == json_resp.json()


def test_stream_invariants_hold(client):
    async def fake_recall(**kwargs):
        await emit("routing", "completed", {"search_type": "GRAPH_COMPLETION"})
        await emit("graph_retrieval", "completed", {"object_count": 2})
        await emit("llm_generation", "started")
        return []

    recall_pkg.recall = fake_recall

    resp = client.post("/recall", json={"query": "q"}, headers={"Accept": NDJSON_MEDIA_TYPE})
    events = _parse_ndjson(resp.text)

    # seq strictly increasing, gap-free, from 1, on every line.
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(1, len(seqs) + 1))

    # Exactly one terminal event, always last.
    terminals = [e for e in events if e["type"] in ("result", "error")]
    assert len(terminals) == 1
    assert events[-1] is terminals[0]

    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "progress"]
    assert ("routing", "completed") in stages
    assert ("llm_generation", "started") in stages


def test_mid_stage_error_leaves_unpaired_started_then_one_terminal_error(client):
    async def fake_recall(**kwargs):
        await emit("llm_generation", "started")  # started with no matching completed
        raise RuntimeError("boom")

    recall_pkg.recall = fake_recall

    resp = client.post("/recall", json={"query": "q"}, headers={"Accept": NDJSON_MEDIA_TYPE})
    events = _parse_ndjson(resp.text)

    progress = [e for e in events if e["type"] == "progress"]
    assert progress == [
        {
            "type": "progress",
            "stage": "llm_generation",
            "status": "started",
            "seq": progress[0]["seq"],
        }
    ]
    # Exactly one terminal error closes the stream.
    assert events[-1]["type"] == "error"
    assert events[-1]["status_code"] == 409
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
