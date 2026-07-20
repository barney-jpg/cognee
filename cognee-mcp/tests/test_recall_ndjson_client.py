"""Tests for CogneeClient.recall NDJSON consumption in API mode (Task 8)."""

import asyncio
import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

cognee_client = importlib.import_module("src.cognee_client")
CogneeClient = cognee_client.CogneeClient
RecallError = cognee_client.RecallError
NDJSON_MEDIA_TYPE = cognee_client.NDJSON_MEDIA_TYPE


class _FakeResponse:
    def __init__(self, headers, lines=None, json_body=None, status_code=200):
        self.headers = headers
        self._lines = lines or []
        self._json = json_body
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        return self._json


class _FakeStreamCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.captured = None

    def stream(self, method, url, json=None, headers=None):
        self.captured = {"method": method, "url": url, "json": json, "headers": headers}
        return _FakeStreamCM(self._response)


def _client_with(response):
    client = CogneeClient(api_url="http://localhost:8000", api_token="token")
    fake = _FakeClient(response)
    client.client = fake
    return client, fake


def test_ndjson_returns_result_and_ignores_progress_keepalive():
    lines = [
        json.dumps({"type": "progress", "stage": "routing", "status": "completed", "seq": 1}),
        json.dumps({"type": "keepalive", "seq": 2, "elapsed_ms": 10000}),
        json.dumps({"type": "result", "data": [{"answer": "hi"}], "seq": 3}),
    ]
    resp = _FakeResponse({"content-type": NDJSON_MEDIA_TYPE}, lines=lines)
    client, fake = _client_with(resp)

    result = asyncio.run(client.recall("q"))
    assert result == [{"answer": "hi"}]
    # Client requested NDJSON.
    assert fake.captured["headers"]["Accept"] == NDJSON_MEDIA_TYPE


def test_ndjson_terminal_error_raises_typed_exception():
    lines = [
        json.dumps({"type": "progress", "stage": "llm_generation", "status": "started", "seq": 1}),
        json.dumps(
            {"type": "error", "status_code": 402, "message": "Token budget exhausted", "seq": 2}
        ),
    ]
    resp = _FakeResponse({"content-type": NDJSON_MEDIA_TYPE}, lines=lines)
    client, _ = _client_with(resp)

    with pytest.raises(RecallError) as excinfo:
        asyncio.run(client.recall("q"))
    assert excinfo.value.status_code == 402
    assert excinfo.value.message == "Token budget exhausted"


def test_ndjson_error_with_stage_preserved():
    lines = [
        json.dumps(
            {
                "type": "error",
                "status_code": 404,
                "message": "Recall prerequisites not met",
                "stage": "graph_retrieval",
                "seq": 1,
            }
        ),
    ]
    resp = _FakeResponse({"content-type": NDJSON_MEDIA_TYPE}, lines=lines)
    client, _ = _client_with(resp)

    with pytest.raises(RecallError) as excinfo:
        asyncio.run(client.recall("q"))
    assert excinfo.value.status_code == 404
    assert excinfo.value.stage == "graph_retrieval"


def test_permission_denied_empty_result_returns_empty_list():
    # Server maps PermissionDenied to a terminal result with empty data — not an error.
    lines = [json.dumps({"type": "result", "data": [], "seq": 1})]
    resp = _FakeResponse({"content-type": NDJSON_MEDIA_TYPE}, lines=lines)
    client, _ = _client_with(resp)

    assert asyncio.run(client.recall("q")) == []


def test_json_fallback_for_old_backend():
    # An older backend ignores Accept and returns a JSON body.
    resp = _FakeResponse({"content-type": "application/json"}, json_body=[{"answer": "legacy"}])
    client, _ = _client_with(resp)

    assert asyncio.run(client.recall("q")) == [{"answer": "legacy"}]
