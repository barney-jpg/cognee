"""Tests for the SLM benchmark comparators.

Pure functions over JSON-shaped values — no network, no model, no cognee state. These
are the tests that let the scoring logic be trusted before a single model call is made.
"""

import pytest

from cognee.eval_framework.slm_task_benchmark.compare import (
    SIMILARITY_SCORE_TOLERANCE,
    chunk_similarity,
    exact_label,
    exact_search_type,
    get_comparator,
    graph_f1,
    node_set_f1,
    normalize_name,
    rouge_l,
    summary_overlap,
)


class TestNormalizeName:
    """Normalization is shared by every set-based comparator, so it is tested alone."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Apollo 11", "apollo 11"),
            ("  Apollo   11.  ", "apollo 11"),
            ("APOLLO-11", "apollo 11"),
            ("Müller", "muller"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalizes_case_punctuation_accents_and_whitespace(self, raw, expected):
        assert normalize_name(raw) == expected


class TestRougeL:
    def test_identical_text_scores_one(self):
        assert rouge_l("the eagle has landed", "the eagle has landed") == 1.0

    def test_disjoint_text_scores_zero(self):
        assert rouge_l("alpha beta", "gamma delta") == 0.0

    def test_two_empty_strings_count_as_agreement(self):
        assert rouge_l("", "") == 1.0

    def test_partial_overlap_scores_between(self):
        score = rouge_l("the eagle has landed", "the eagle landed")
        assert 0.0 < score < 1.0


class TestExactSearchType:
    def test_same_type_in_different_case_agrees(self):
        result = exact_search_type("GRAPH_COMPLETION", "graph_completion")
        assert result.score == 1.0

    def test_different_type_disagrees(self):
        result = exact_search_type("GRAPH_COMPLETION", "CHUNKS")
        assert result.score == 0.0
        assert result.detail["candidate_is_valid_search_type"] is True

    def test_invented_type_is_flagged_as_invalid(self):
        """A hallucinated type is wrong *and* structurally invalid — both are recorded."""
        result = exact_search_type("GRAPH_COMPLETION", "MAGIC_SEARCH")
        assert result.score == 0.0
        assert result.detail["candidate_is_valid_search_type"] is False

    def test_missing_candidate_scores_zero(self):
        result = exact_search_type("GRAPH_COMPLETION", None)
        assert result.score == 0.0
        assert result.detail["reason"] == "candidate missing"


class TestExactLabel:
    def _prediction(self, content_type: str, subclasses: list[str]) -> dict:
        return {"label": {"type": content_type, "subclass": subclasses}}

    def test_identical_prediction_scores_one(self):
        prediction = self._prediction("TEXT", ["News stories and blog posts"])
        assert exact_label(prediction, prediction).score == 1.0

    def test_wrong_type_scores_zero_even_with_matching_subclass(self):
        result = exact_label(
            self._prediction("TEXT", ["News stories and blog posts"]),
            self._prediction("AUDIO", ["News stories and blog posts"]),
        )
        assert result.score == 0.0
        assert result.detail["type_matches"] is False

    def test_right_type_wrong_subclass_scores_half(self):
        """Type dominates; a neighbouring subclass is a smaller error than a wrong type."""
        result = exact_label(
            self._prediction("TEXT", ["News stories and blog posts"]),
            self._prediction("TEXT", ["Books and manuscripts"]),
        )
        assert result.score == 0.5


class TestChunkSimilarity:
    def _similarity(self, are_similar: bool, score: float) -> dict:
        return {"are_similar": are_similar, "similarity_score": score, "reasoning": "because"}

    def test_matching_verdict_scores_one(self):
        result = chunk_similarity(self._similarity(True, 0.9), self._similarity(True, 0.8))
        assert result.score == 1.0
        assert result.detail["score_within_tolerance"] is True

    def test_opposite_verdict_scores_zero(self):
        result = chunk_similarity(self._similarity(True, 0.9), self._similarity(False, 0.1))
        assert result.score == 0.0

    def test_score_delta_beyond_tolerance_is_reported(self):
        """The verdict still carries the score; the float delta is reported separately."""
        delta = SIMILARITY_SCORE_TOLERANCE + 0.1
        result = chunk_similarity(self._similarity(True, 0.9), self._similarity(True, 0.9 - delta))
        assert result.score == 1.0
        assert result.detail["score_within_tolerance"] is False

    def test_out_of_range_score_is_flagged(self):
        """The ge=0/le=1 constraint is exactly what small models violate."""
        result = chunk_similarity(self._similarity(True, 0.9), self._similarity(True, 1.7))
        assert result.detail["candidate_score_in_range"] is False


class TestNodeSetF1:
    def test_identical_node_sets_score_one(self):
        nodes = {"nodes": ["Apollo 11", "Neil Armstrong"]}
        assert node_set_f1(nodes, nodes).score == 1.0

    def test_normalization_makes_formatting_differences_agree(self):
        result = node_set_f1({"nodes": ["Apollo 11"]}, {"nodes": ["  apollo-11. "]})
        assert result.score == 1.0

    def test_partial_overlap_scores_between(self):
        result = node_set_f1(
            {"nodes": ["Apollo 11", "Neil Armstrong"]},
            {"nodes": ["Apollo 11", "Buzz Aldrin", "NASA"]},
        )
        assert 0.0 < result.score < 1.0
        assert result.detail["candidate_count"] == 3

    def test_two_empty_lists_count_as_agreement(self):
        """Both models correctly finding nothing is agreement, not a zero."""
        assert node_set_f1({"nodes": []}, {"nodes": []}).score == 1.0

    def test_empty_candidate_against_populated_reference_scores_zero(self):
        assert node_set_f1({"nodes": ["Apollo 11"]}, {"nodes": []}).score == 0.0


class TestSummaryOverlap:
    def test_identical_summary_scores_one(self):
        summary = {"summary": "Apollo 11 landed on the moon in 1969."}
        assert summary_overlap(summary, summary).score == 1.0

    def test_truncated_summary_is_visible_in_the_length_ratio(self):
        result = summary_overlap(
            {"summary": "Apollo 11 landed on the moon in July 1969 carrying three astronauts."},
            {"summary": "Apollo 11 landed"},
        )
        assert result.detail["length_ratio"] < 0.5


class TestGraphF1:
    def _graph(self) -> dict:
        return {
            "nodes": [
                {"id": "n1", "name": "Apollo 11", "type": "MISSION", "description": ""},
                {"id": "n2", "name": "Neil Armstrong", "type": "PERSON", "description": ""},
            ],
            "edges": [
                {
                    "source_node_id": "n2",
                    "target_node_id": "n1",
                    "relationship_name": "commanded",
                }
            ],
        }

    def test_identical_graph_scores_one(self):
        graph = self._graph()
        result = graph_f1(graph, graph)
        assert result.score == 1.0
        assert result.detail["node_f1"] == 1.0
        assert result.detail["edge_f1"] == 1.0

    def test_edges_are_matched_by_node_name_not_by_id(self):
        """Two models never agree on ids; resolving through names is what makes graphs
        comparable at all."""
        candidate = self._graph()
        candidate["nodes"][0]["id"] = "x9"
        candidate["nodes"][1]["id"] = "x7"
        candidate["edges"][0] = {
            "source_node_id": "x7",
            "target_node_id": "x9",
            "relationship_name": "commanded",
        }
        assert graph_f1(self._graph(), candidate).score == 1.0

    def test_right_nodes_but_invented_edge_lowers_only_edge_f1(self):
        candidate = self._graph()
        candidate["edges"][0]["relationship_name"] = "was born on"
        result = graph_f1(self._graph(), candidate)
        assert result.detail["node_f1"] == 1.0
        assert result.detail["edge_f1"] == 0.0


class TestComparatorRegistry:
    def test_every_task_names_a_registered_comparator(self):
        """Guards against a task record pointing at a comparator that does not exist."""
        from cognee.eval_framework.slm_task_benchmark.tasks import BENCHMARK_TASKS

        for task in BENCHMARK_TASKS.values():
            assert get_comparator(task.comparator) is not None

    def test_unknown_comparator_raises_with_the_available_names(self):
        with pytest.raises(ValueError, match="Unknown comparator"):
            get_comparator("nope")
