"""Tests for CogneeClient.recall's only_context pass-through (issue #13).

only_context=True asks the backend for the assembled context without a
synthesized answer. The flag has to survive both transports — the HTTP payload
in API mode and the kwargs in SDK mode — because the MCP server is the only
caller and picks its transport at construction time.

Every test passes an explicit ``datasets`` argument. A bare recall with neither
datasets nor session_id takes the list_datasets() fallback path, which is not
what these tests are about.
"""

import asyncio
import importlib
import json
import sys
from pathlib import Path

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

cognee_client = importlib.import_module("src.cognee_client")
CogneeClient = cognee_client.CogneeClient
NDJSON_MEDIA_TYPE = cognee_client.NDJSON_MEDIA_TYPE


class _FakeResponse:
    """Minimal NDJSON response carrying a single terminal result event."""

    def __init__(self):
        self.headers = {"content-type": NDJSON_MEDIA_TYPE}

    async def aiter_lines(self):
        yield json.dumps({"type": "result", "data": [{"answer": "ok"}], "seq": 1})


class _FakeHttpClient:
    """Captures the request the client would have sent."""

    def __init__(self):
        self.captured = None

    def stream(self, method, url, json=None, headers=None):
        self.captured = {"method": method, "url": url, "json": json, "headers": headers}
        return _FakeStreamCM(_FakeResponse())


class _FakeStreamCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeCognee:
    """Stands in for the cognee SDK module in direct mode."""

    def __init__(self):
        self.captured = None

    async def recall(self, **kwargs):
        self.captured = kwargs
        return [{"answer": "ok"}]


def _api_mode_client() -> tuple:
    client = CogneeClient(api_url="http://localhost:8000", api_token="token")
    fake = _FakeHttpClient()
    client.client = fake
    return client, fake


def _sdk_mode_client() -> tuple:
    """Build an API-mode client, then flip it to direct mode with a fake SDK.

    Constructing directly would import the real cognee package; swapping the two
    attributes the direct path reads keeps this a unit test.
    """
    client = CogneeClient(api_url="http://localhost:8000", api_token="token")
    fake = _FakeCognee()
    client.use_api = False
    client.cognee = fake
    return client, fake


class TestApiMode:
    def test_only_context_is_sent_in_the_payload(self):
        client, fake = _api_mode_client()

        asyncio.run(client.recall("q", datasets=["ds"], only_context=True))

        assert fake.captured["json"]["only_context"] is True

    def test_defaults_to_false(self):
        """Recall must keep synthesizing an answer unless a caller opts out."""
        client, fake = _api_mode_client()

        asyncio.run(client.recall("q", datasets=["ds"]))

        assert fake.captured["json"]["only_context"] is False


class TestSdkMode:
    def test_only_context_is_forwarded_to_the_sdk(self):
        client, fake = _sdk_mode_client()

        asyncio.run(client.recall("q", datasets=["ds"], only_context=True))

        assert fake.captured["only_context"] is True

    def test_defaults_to_false(self):
        client, fake = _sdk_mode_client()

        asyncio.run(client.recall("q", datasets=["ds"]))

        assert fake.captured["only_context"] is False
