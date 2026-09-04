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


# Datasets the fake backend reports. A bare recall (no datasets, no session_id)
# takes recall()'s fallback path and asks for these before it starts streaming,
# so the fake has to answer that request or the stream is never reached.
_VISIBLE_DATASETS = [
    {"id": "d1", "name": "alpha", "created_at": "2026-01-01"},
    {"id": "d2", "name": "beta", "created_at": "2026-01-02"},
]


class _FakeClient:
    def __init__(self, response, datasets=None):
        self._response = response
        self._datasets = _VISIBLE_DATASETS if datasets is None else datasets
        self.captured = None
        self.dataset_requests = 0

    async def get(self, url, headers=None):
        """Serve list_datasets(); recall()'s no-scope fallback calls this first."""
        self.dataset_requests += 1
        return _FakeResponse({}, json_body=self._datasets)

    def stream(self, method, url, json=None, headers=None):
        self.captured = {"method": method, "url": url, "json": json, "headers": headers}
        return _FakeStreamCM(self._response)


def _client_with(response, datasets=None):
    client = CogneeClient(api_url="http://localhost:8000", api_token="token")
    fake = _FakeClient(response, datasets=datasets)
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


def test_truncated_stream_raises():
    # Stream ends without any terminal event (proxy reset / crash): must fail loud, not return
    # a silent None success.
    lines = [json.dumps({"type": "progress", "stage": "routing", "status": "completed", "seq": 1})]
    resp = _FakeResponse({"content-type": NDJSON_MEDIA_TYPE}, lines=lines)
    client, _ = _client_with(resp)

    with pytest.raises(RecallError):
        asyncio.run(client.recall("q"))


def test_terminal_result_missing_data_raises():
    lines = [json.dumps({"type": "result", "seq": 1})]  # malformed terminal: no "data"
    resp = _FakeResponse({"content-type": NDJSON_MEDIA_TYPE}, lines=lines)
    client, _ = _client_with(resp)

    with pytest.raises(RecallError):
        asyncio.run(client.recall("q"))


def test_multiple_terminals_raise():
    lines = [
        json.dumps({"type": "result", "data": [1], "seq": 1}),
        json.dumps({"type": "result", "data": [2], "seq": 2}),
    ]
    resp = _FakeResponse({"content-type": NDJSON_MEDIA_TYPE}, lines=lines)
    client, _ = _client_with(resp)

    with pytest.raises(RecallError):
        asyncio.run(client.recall("q"))


def test_content_type_is_case_insensitive_with_charset():
    # A valid, differently-cased media type (with a charset param) must still stream, not fall
    # through to the legacy JSON path.
    lines = [json.dumps({"type": "result", "data": [{"ok": True}], "seq": 1})]
    resp = _FakeResponse({"content-type": "Application/X-NDJSON; charset=utf-8"}, lines=lines)
    client, _ = _client_with(resp)

    assert asyncio.run(client.recall("q")) == [{"ok": True}]


# --- no-scope fallback -------------------------------------------------------
# recall() widens a bare call to every visible dataset, because an unscoped
# recall would otherwise hit the empty default dataset and 404. That behaviour
# is what made this file's fake insufficient in the first place, so it is
# pinned here rather than left implicit.


def _ok_response():
    return _FakeResponse(
        {"content-type": NDJSON_MEDIA_TYPE},
        lines=[json.dumps({"type": "result", "data": [{"ok": True}], "seq": 1})],
    )


def test_bare_recall_widens_to_every_visible_dataset():
    client, fake = _client_with(_ok_response())

    asyncio.run(client.recall("q"))

    assert fake.dataset_requests == 1
    assert fake.captured["json"]["datasets"] == ["alpha", "beta"]


def test_explicit_datasets_skip_the_lookup():
    client, fake = _client_with(_ok_response())

    asyncio.run(client.recall("q", datasets=["chosen"]))

    assert fake.dataset_requests == 0
    assert fake.captured["json"]["datasets"] == ["chosen"]


def test_session_scoped_recall_is_left_unscoped():
    """A session recall must stay unscoped so it can search the session cache."""
    client, fake = _client_with(_ok_response())

    asyncio.run(client.recall("q", session_id="s1"))

    assert fake.dataset_requests == 0
    assert "datasets" not in fake.captured["json"]
    assert fake.captured["json"]["session_id"] == "s1"


def test_backend_reporting_no_datasets_leaves_the_call_unscoped():
    """An empty dataset list must not add an empty `datasets` key to the payload."""
    client, fake = _client_with(_ok_response(), datasets=[])

    asyncio.run(client.recall("q"))

    assert fake.dataset_requests == 1
    assert "datasets" not in fake.captured["json"]
