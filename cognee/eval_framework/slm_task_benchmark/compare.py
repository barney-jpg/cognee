"""Comparators: how close is a small model's output to the reference model's?

Every comparator takes **plain JSON-compatible values**, not Pydantic instances, so a
report can be regenerated from a stored result file without reconstructing any models.
Each returns a headline ``score`` in [0, 1] plus a ``detail`` breakdown.

Normalization lives here and nowhere else. The eval harness in the SLM/LoRA epic
(sub-issue "evaluation harness") has to score entity names the same way; when that lands,
it should import these helpers rather than restate the rules.

Design note: this is a Strategy registry — ``COMPARATORS`` maps a task's ``comparator``
name to a function, so the runner and the report never branch on the task.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

# Score at or above which two float scores are treated as agreeing (see chunk_similarity).
SIMILARITY_SCORE_TOLERANCE = 0.2

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing one candidate output against the reference output."""

    score: float
    detail: dict[str, Any] = field(default_factory=dict)


def normalize_name(value: Any) -> str:
    """Canonical form for comparing names and labels.

    Lowercases, strips accents and punctuation, collapses whitespace. Applied to entity
    names, node names and relationship names alike so boundary noise ("Apollo 11." vs
    "apollo 11") does not count as a disagreement.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _PUNCTUATION.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()


def _tokenize(value: Any) -> list[str]:
    normalized = normalize_name(value)
    return normalized.split() if normalized else []


def _set_f1(reference: set[str], candidate: set[str]) -> dict[str, float]:
    """Precision / recall / F1 over two sets, with both-empty counted as perfect."""
    if not reference and not candidate:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = len(reference & candidate)
    precision = overlap / len(candidate) if candidate else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Length of the longest common subsequence — the basis of ROUGE-L."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right):
            if left_token == right_token:
                current.append(previous[column] + 1)
            else:
                current.append(max(current[column], previous[column + 1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F-measure. Implemented here to keep the package dependency-free."""
    reference_tokens = _tokenize(reference)
    candidate_tokens = _tokenize(candidate)
    if not reference_tokens and not candidate_tokens:
        return 1.0
    overlap = _lcs_length(reference_tokens, candidate_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _missing(reference: Any, candidate: Any) -> ComparisonResult | None:
    """A hard failure on either side scores zero — there is nothing to compare."""
    if reference is None or candidate is None:
        return ComparisonResult(
            score=0.0,
            detail={"reason": "reference missing" if reference is None else "candidate missing"},
        )
    return None


def exact_search_type(reference: Any, candidate: Any) -> ComparisonResult:
    """Search-type routing: the raw string must name the same SearchType.

    Production upper-cases the answer and checks membership in the enum
    (select_search_type.py:37), so an answer that is not a valid member is as wrong as
    the wrong member — recorded separately because it is a different failure mode.
    """
    if (early := _missing(reference, candidate)) is not None:
        return early

    from cognee.modules.search.types import SearchType

    reference_type = str(reference).strip().upper()
    candidate_type = str(candidate).strip().upper()
    candidate_is_valid = candidate_type in SearchType.__members__
    agrees = candidate_is_valid and candidate_type == reference_type
    return ComparisonResult(
        score=1.0 if agrees else 0.0,
        detail={
            "reference": reference_type,
            "candidate": candidate_type,
            "candidate_is_valid_search_type": candidate_is_valid,
        },
    )


def exact_label(reference: Any, candidate: Any) -> ComparisonResult:
    """Content classification: same content type, and overlapping subclasses.

    Type and subclass are scored separately: picking the right type but a neighbouring
    subclass is a much smaller error than misreading the content entirely.
    """
    if (early := _missing(reference, candidate)) is not None:
        return early

    reference_label = (reference or {}).get("label") or {}
    candidate_label = (candidate or {}).get("label") or {}

    type_matches = normalize_name(reference_label.get("type")) == normalize_name(
        candidate_label.get("type")
    )
    subclass_scores = _set_f1(
        {normalize_name(item) for item in reference_label.get("subclass") or []},
        {normalize_name(item) for item in candidate_label.get("subclass") or []},
    )
    # Type dominates: a wrong type makes the subclass meaningless.
    score = 0.0 if not type_matches else 0.5 + 0.5 * subclass_scores["f1"]
    return ComparisonResult(
        score=score,
        detail={
            "type_matches": type_matches,
            "reference_type": reference_label.get("type"),
            "candidate_type": candidate_label.get("type"),
            "subclass_f1": subclass_scores["f1"],
        },
    )


def chunk_similarity(reference: Any, candidate: Any) -> ComparisonResult:
    """Chunk association: the boolean verdict must match, the score must be close.

    The boolean is what the production code acts on, so it carries the score. The float
    is reported alongside because it is the field with the ``ge=0 le=1`` constraint —
    the part of the schema small models are most likely to violate.
    """
    if (early := _missing(reference, candidate)) is not None:
        return early

    reference_verdict = bool(reference.get("are_similar"))
    candidate_verdict = bool(candidate.get("are_similar"))
    verdict_matches = reference_verdict == candidate_verdict

    reference_score = reference.get("similarity_score")
    candidate_score = candidate.get("similarity_score")
    score_delta: float | None = None
    score_within_tolerance = False
    if isinstance(reference_score, (int, float)) and isinstance(candidate_score, (int, float)):
        score_delta = abs(float(reference_score) - float(candidate_score))
        score_within_tolerance = score_delta <= SIMILARITY_SCORE_TOLERANCE

    candidate_score_in_range = (
        isinstance(candidate_score, (int, float)) and 0.0 <= float(candidate_score) <= 1.0
    )

    return ComparisonResult(
        score=1.0 if verdict_matches else 0.0,
        detail={
            "verdict_matches": verdict_matches,
            "score_delta": score_delta,
            "score_within_tolerance": score_within_tolerance,
            "candidate_score_in_range": candidate_score_in_range,
        },
    )


def node_set_f1(reference: Any, candidate: Any) -> ComparisonResult:
    """Cascade node extraction: set F1 over normalized node names."""
    if (early := _missing(reference, candidate)) is not None:
        return early

    reference_nodes = {normalize_name(node) for node in reference.get("nodes") or []}
    candidate_nodes = {normalize_name(node) for node in candidate.get("nodes") or []}
    reference_nodes.discard("")
    candidate_nodes.discard("")

    scores = _set_f1(reference_nodes, candidate_nodes)
    return ComparisonResult(
        score=scores["f1"],
        detail={
            **scores,
            "reference_count": len(reference_nodes),
            "candidate_count": len(candidate_nodes),
        },
    )


def summary_overlap(reference: Any, candidate: Any) -> ComparisonResult:
    """Summarization: ROUGE-L against the reference summary, plus a length ratio.

    Free text has no exact match. The length ratio is reported because the characteristic
    small-model failure here is not a wrong summary but a truncated or runaway one.
    """
    if (early := _missing(reference, candidate)) is not None:
        return early

    reference_summary = str(reference.get("summary") or "")
    candidate_summary = str(candidate.get("summary") or "")
    overlap = rouge_l(reference_summary, candidate_summary)

    reference_length = len(_tokenize(reference_summary))
    candidate_length = len(_tokenize(candidate_summary))
    length_ratio = candidate_length / reference_length if reference_length else 0.0

    return ComparisonResult(
        score=overlap,
        detail={
            "rouge_l": overlap,
            "length_ratio": length_ratio,
            "reference_tokens": reference_length,
            "candidate_tokens": candidate_length,
        },
    )


def _edge_triples(graph: dict[str, Any]) -> set[str]:
    """Edges as (source name, relationship, target name), resolved through the node list.

    Edges reference node *ids*, which differ between models even for identical graphs.
    Resolving to names first is what makes two graphs comparable at all.
    """
    names_by_id = {
        str(node.get("id")): normalize_name(node.get("name") or node.get("id"))
        for node in graph.get("nodes") or []
    }
    triples = set()
    for edge in graph.get("edges") or []:
        source = names_by_id.get(str(edge.get("source_node_id")), "")
        target = names_by_id.get(str(edge.get("target_node_id")), "")
        relationship = normalize_name(edge.get("relationship_name"))
        if source and target and relationship:
            triples.add(f"{source}|{relationship}|{target}")
    return triples


def graph_f1(reference: Any, candidate: Any) -> ComparisonResult:
    """Graph extraction (control task): node F1 and edge F1, reported separately.

    Kept separate on purpose — a model can name the right entities and still invent the
    relationships between them, and that distinction is the whole point of the control.
    """
    if (early := _missing(reference, candidate)) is not None:
        return early

    reference_nodes = {
        normalize_name(node.get("name") or node.get("id")) for node in reference.get("nodes") or []
    }
    candidate_nodes = {
        normalize_name(node.get("name") or node.get("id")) for node in candidate.get("nodes") or []
    }
    reference_nodes.discard("")
    candidate_nodes.discard("")

    node_scores = _set_f1(reference_nodes, candidate_nodes)
    edge_scores = _set_f1(_edge_triples(reference), _edge_triples(candidate))

    return ComparisonResult(
        score=(node_scores["f1"] + edge_scores["f1"]) / 2,
        detail={
            "node_f1": node_scores["f1"],
            "edge_f1": edge_scores["f1"],
            "reference_node_count": len(reference_nodes),
            "candidate_node_count": len(candidate_nodes),
        },
    )


COMPARATORS: dict[str, Callable[[Any, Any], ComparisonResult]] = {
    "exact_search_type": exact_search_type,
    "exact_label": exact_label,
    "chunk_similarity": chunk_similarity,
    "node_set_f1": node_set_f1,
    "summary_overlap": summary_overlap,
    "graph_f1": graph_f1,
}


def get_comparator(name: str) -> Callable[[Any, Any], ComparisonResult]:
    """Look up a comparator by name, with a helpful error listing the valid names."""
    try:
        return COMPARATORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown comparator {name!r}. Available: {', '.join(COMPARATORS)}"
        ) from None
