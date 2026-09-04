"""Emit-point tests for get_retriever_output (Task 6).

Verifies graph_retrieval / llm_generation stage events fire at the right boundaries, are
skipped on the early-answer and only_context paths, and that failures attach `stage` for
terminal error attribution.
"""

import importlib
from types import SimpleNamespace

import pytest

from cognee.modules.recall.progress import progress_scope
from cognee.modules.search.types import SearchType

# Import the module object (not the same-named function) so monkeypatch targets its globals.
gro = importlib.import_module("cognee.modules.search.methods.get_retriever_output")
# The stage boundaries live here: retrieval, context extraction and completion were extracted
# out of get_retriever_output into this module, so that is where the emits fire.
sac = importlib.import_module("cognee.modules.retrieval.session_aware_completion")


class _FakeRetriever:
    def __init__(self, should_answer=True, fail_objects=False, fail_completion=False):
        self._should_answer = should_answer
        self._fail_objects = fail_objects
        self._fail_completion = fail_completion

    async def prepare_session_turn_for_retrieval(self, query):
        return SimpleNamespace(
            should_answer=self._should_answer,
            effective_query=query,
            response_to_user="Got it." if not self._should_answer else None,
        )

    async def get_retrieved_objects(self, query):
        if self._fail_objects:
            raise RuntimeError("retrieval boom")
        return [1, 2, 3]

    async def get_context_from_objects(self, query, retrieved_objects):
        return "context"

    async def get_completion_from_context(self, **kwargs):
        if self._fail_completion:
            raise RuntimeError("completion boom")
        return "answer"


@pytest.fixture
def patch_retriever(monkeypatch):
    def _install(retriever):
        async def _fake_graph_engine():
            return SimpleNamespace(is_empty=_async_return(False))

        async def _fake_get_instance(**kwargs):
            return retriever

        async def _noop_access(objs):
            return None

        monkeypatch.setattr(gro, "get_graph_engine", _fake_graph_engine)
        monkeypatch.setattr(gro, "get_search_type_retriever_instance", _fake_get_instance)
        monkeypatch.setattr(sac, "update_node_access_timestamps", _noop_access)

    return _install


def _async_return(value):
    async def _inner():
        return value

    return _inner


async def _drain(emitter):
    events = []
    while not emitter.queue.empty():
        events.append(emitter.queue.get_nowait())
    return events


@pytest.mark.asyncio
async def test_emits_graph_retrieval_then_llm_generation(patch_retriever):
    patch_retriever(_FakeRetriever())
    async with progress_scope() as emitter:
        await gro.get_retriever_output(SearchType.GRAPH_COMPLETION, "q")
        events = await _drain(emitter)

    stages = [(e["stage"], e["status"]) for e in events]
    assert stages == [("graph_retrieval", "completed"), ("llm_generation", "started")]
    assert events[0]["detail"]["object_count"] == 3


@pytest.mark.asyncio
async def test_only_context_skips_llm_generation(patch_retriever):
    patch_retriever(_FakeRetriever())
    async with progress_scope() as emitter:
        await gro.get_retriever_output(SearchType.GRAPH_COMPLETION, "q", only_context=True)
        events = await _drain(emitter)

    stages = [e["stage"] for e in events]
    assert stages == ["graph_retrieval"]  # no llm_generation


@pytest.mark.asyncio
async def test_early_answer_emits_neither(patch_retriever):
    patch_retriever(_FakeRetriever(should_answer=False))
    async with progress_scope() as emitter:
        await gro.get_retriever_output(SearchType.GRAPH_COMPLETION, "q")
        events = await _drain(emitter)

    assert events == []  # short-circuit before retrieval + completion


@pytest.mark.asyncio
async def test_graph_retrieval_failure_attaches_stage(patch_retriever):
    patch_retriever(_FakeRetriever(fail_objects=True))
    async with progress_scope():
        with pytest.raises(RuntimeError) as excinfo:
            await gro.get_retriever_output(SearchType.GRAPH_COMPLETION, "q")
    assert getattr(excinfo.value, "stage", None) == "graph_retrieval"


@pytest.mark.asyncio
async def test_llm_generation_failure_attaches_stage(patch_retriever):
    patch_retriever(_FakeRetriever(fail_completion=True))
    async with progress_scope():
        with pytest.raises(RuntimeError) as excinfo:
            await gro.get_retriever_output(SearchType.GRAPH_COMPLETION, "q")
    assert getattr(excinfo.value, "stage", None) == "llm_generation"
