"""Content-negotiation tests for the recall endpoint (Task 4).

Covers the Accept-parsing matrix at the unit level plus endpoint-level assertions that the
default JSON representation is unchanged and NDJSON is served only on explicit opt-in.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cognee.api.v1.recall as recall_pkg
from cognee.api.v1.recall.routers.get_recall_router import get_recall_router
from cognee.api.v1.recall.stream import NDJSON_MEDIA_TYPE, accept_prefers_ndjson
from cognee.modules.users.methods import get_authenticated_user

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())


@pytest.mark.parametrize(
    "accept,expected",
    [
        (None, False),
        ("", False),
        ("*/*", False),
        ("application/json", False),
        ("text/event-stream", False),
        ("application/x-ndjson", True),
        ("application/x-ndjson;q=1", True),
        ("application/x-ndjson;q=0", False),
        ("application/x-ndjson; q=0.0", False),
        ("application/json, application/x-ndjson", True),
        ("application/json;q=0.9, application/x-ndjson;q=0.8", True),
        ("APPLICATION/X-NDJSON", True),  # case-insensitive media type
        ("application/x-ndjson;q=bogus", False),  # malformed q -> treated as 0
    ],
)
def test_accept_prefers_ndjson_matrix(accept, expected):
    assert accept_prefers_ndjson(accept) is expected


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_recall_router(), prefix="/recall")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


def test_default_is_json_unchanged(client):
    recall_pkg.recall = AsyncMock(return_value=[])
    resp = client.post("/recall", json={"query": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == []


def test_ndjson_accept_streams(client):
    recall_pkg.recall = AsyncMock(return_value=[])
    resp = client.post("/recall", json={"query": "hi"}, headers={"Accept": NDJSON_MEDIA_TYPE})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(NDJSON_MEDIA_TYPE)
    assert resp.headers.get("x-accel-buffering") == "no"
    assert resp.headers.get("cache-control") == "no-cache"

    lines = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert lines[-1]["type"] == "result"
    assert lines[-1]["data"] == []


def test_q_zero_falls_back_to_json(client):
    recall_pkg.recall = AsyncMock(return_value=[])
    resp = client.post(
        "/recall", json={"query": "hi"}, headers={"Accept": "application/x-ndjson;q=0"}
    )
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == []
