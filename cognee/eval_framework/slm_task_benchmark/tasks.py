"""Registry of the cognee call sites this benchmark measures.

Each entry is a data record describing one real ``LLMGateway.acreate_structured_output``
call site: which prompts it uses, which Pydantic model it expects back, and how two
outputs are compared. The runner iterates over these records without knowing anything
about the individual tasks — adding a task means adding a record here, not a branch in
the runner. (Registry pattern; the per-task ``build_prompts`` callable is the Strategy
that keeps the runner generic.)

The prompts and response models are **imported from the production code**, never
re-typed. A benchmark that measures a paraphrased prompt measures the wrong thing.

The tasks are ordered by schema complexity so the results show *where* small models stop
coping, rather than only whether they do.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.root_dir import get_absolute_path
from cognee.shared.data_models import (
    DefaultContentPrediction,
    KnowledgeGraph,
    SummarizedContent,
)
from cognee.tasks.chunks.create_chunk_associations import ChunkSimilarity
from cognee.tasks.graph.cascade_extract.utils.extract_nodes import PotentialNodes

_CASCADE_PROMPT_DIRECTORY = "./tasks/graph/cascade_extract/prompts"


class InputKind(str, Enum):
    """What kind of sample a task consumes. Decides which corpus sampler feeds it."""

    QUERY = "query"
    CHUNK = "chunk"
    CHUNK_PAIR = "chunk_pair"


@dataclass(frozen=True)
class BenchmarkTask:
    """One measured call site.

    Attributes:
        key: Stable identifier used on the CLI and in result files.
        title: Human-readable name for the report.
        call_site: Where this runs in production, for traceability in the report.
        schema_class: Position on the complexity ladder (none | flat | constrained | ...).
        input_kind: Which sampler provides its input.
        response_model: The Pydantic model the production call site asks for, or ``str``.
        build_prompts: Sample -> (system_prompt, user_prompt), mirroring the call site.
        comparator: Name of the comparator in ``compare.py`` used to score an output.
        on_ladder: False marks a control task that is expected to fail.
    """

    key: str
    title: str
    call_site: str
    schema_class: str
    input_kind: InputKind
    response_model: type
    build_prompts: Callable[[Any], tuple[str, str]]
    comparator: str
    on_ladder: bool = True


def _prompts_search_type(sample: str) -> tuple[str, str]:
    """Mirrors select_search_type (modules/search/operations/select_search_type.py:31).

    The production call passes the raw query as ``text_input`` and asks for a plain
    ``str`` — no schema at all, the easiest rung on the ladder.
    """
    system_prompt = read_query_prompt("search_type_selector_prompt.txt") or ""
    return system_prompt, sample


def _prompts_categories(sample: str) -> tuple[str, str]:
    """Mirrors extract_categories (infrastructure/llm/extraction/extract_categories.py:10).

    The production function takes the response model from its caller. Nothing in the
    codebase calls it today, so we use ``DefaultContentPrediction`` — the model that
    matches what ``classify_content.txt`` actually asks the model to produce.
    """
    system_prompt = read_query_prompt("classify_content.txt") or ""
    return system_prompt, sample


def _prompts_chunk_association(sample: tuple[str, str]) -> tuple[str, str]:
    """Mirrors _compare_chunks (tasks/chunks/create_chunk_associations.py:62)."""
    chunk_1, chunk_2 = sample
    system_prompt = read_query_prompt("chunk_association_system.txt") or ""
    user_prompt = render_prompt(
        "chunk_association_user.txt", {"chunk_1": chunk_1, "chunk_2": chunk_2}
    )
    return system_prompt, user_prompt


def _prompts_cascade_nodes(sample: str) -> tuple[str, str]:
    """Mirrors extract_nodes (tasks/graph/cascade_extract/utils/extract_nodes.py:34).

    Production runs two rounds and feeds round one's nodes back in. We measure a single
    round with an empty ``previous_nodes`` list: the second round's input depends on the
    first round's output, which would make results incomparable across models.
    """
    base_directory = get_absolute_path(_CASCADE_PROMPT_DIRECTORY)
    system_prompt = (
        read_query_prompt("extract_graph_nodes_prompt_system.txt", base_directory=base_directory)
        or ""
    )
    user_prompt = render_prompt(
        "extract_graph_nodes_prompt_input.txt",
        {"previous_nodes": [], "round_number": 1, "total_rounds": 1, "text": sample},
        base_directory=base_directory,
    )
    return system_prompt, user_prompt


def _prompts_summary(sample: str) -> tuple[str, str]:
    """Mirrors extract_summary (infrastructure/llm/extraction/extract_summary.py:29).

    ``SummarizedContent`` is the model the summarization pipeline uses by default
    (modules/cognify/config.py:10).
    """
    system_prompt = read_query_prompt("summarize_content.txt") or ""
    return system_prompt, sample


def _prompts_content_graph(sample: str) -> tuple[str, str]:
    """Mirrors extract_content_graph (infrastructure/llm/extraction/knowledge_graph/
    extract_content_graph.py:39) — the control task.

    The prompt file is read from the live LLM config so the benchmark follows any
    project-level override rather than hardcoding the default.
    """
    from cognee.infrastructure.llm.config import get_llm_config

    system_prompt = render_prompt(get_llm_config().graph_prompt_path, {})
    return system_prompt, sample


BENCHMARK_TASKS: dict[str, BenchmarkTask] = {
    task.key: task
    for task in (
        BenchmarkTask(
            key="search_type",
            title="Search-type routing",
            call_site="modules/search/operations/select_search_type.py:31",
            schema_class="none (plain str)",
            input_kind=InputKind.QUERY,
            response_model=str,
            build_prompts=_prompts_search_type,
            comparator="exact_search_type",
        ),
        BenchmarkTask(
            key="categories",
            title="Content classification",
            call_site="infrastructure/llm/extraction/extract_categories.py:10",
            schema_class="flat (nested union of enums)",
            input_kind=InputKind.CHUNK,
            response_model=DefaultContentPrediction,
            build_prompts=_prompts_categories,
            comparator="exact_label",
        ),
        BenchmarkTask(
            key="chunk_association",
            title="Chunk association",
            call_site="tasks/chunks/create_chunk_associations.py:62",
            schema_class="flat with numeric constraints (float ge=0 le=1)",
            input_kind=InputKind.CHUNK_PAIR,
            response_model=ChunkSimilarity,
            build_prompts=_prompts_chunk_association,
            comparator="chunk_similarity",
        ),
        BenchmarkTask(
            key="cascade_nodes",
            title="Cascade node extraction",
            call_site="tasks/graph/cascade_extract/utils/extract_nodes.py:34",
            schema_class="list of strings",
            input_kind=InputKind.CHUNK,
            response_model=PotentialNodes,
            build_prompts=_prompts_cascade_nodes,
            comparator="node_set_f1",
        ),
        BenchmarkTask(
            key="summary",
            title="Chunk summarization",
            call_site="infrastructure/llm/extraction/extract_summary.py:29",
            schema_class="flat, long free text",
            input_kind=InputKind.CHUNK,
            response_model=SummarizedContent,
            build_prompts=_prompts_summary,
            comparator="summary_overlap",
        ),
        BenchmarkTask(
            key="content_graph",
            title="Graph extraction (control)",
            call_site="infrastructure/llm/extraction/knowledge_graph/extract_content_graph.py:39",
            schema_class="nested (nodes + edges)",
            input_kind=InputKind.CHUNK,
            response_model=KnowledgeGraph,
            build_prompts=_prompts_content_graph,
            comparator="graph_f1",
            on_ladder=False,
        ),
    )
}

LADDER_TASK_KEYS = [key for key, task in BENCHMARK_TASKS.items() if task.on_ladder]
CONTROL_TASK_KEYS = [key for key, task in BENCHMARK_TASKS.items() if not task.on_ladder]


def get_task(key: str) -> BenchmarkTask:
    """Look up a task by key, with a helpful error listing the valid keys."""
    try:
        return BENCHMARK_TASKS[key]
    except KeyError:
        available = ", ".join(BENCHMARK_TASKS)
        raise ValueError(f"Unknown task {key!r}. Available: {available}") from None
