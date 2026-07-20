"""
Unit tests for retrieval metrics.

Pure arithmetic, checked against hand-computed values rather than against the
implementation. A metric that is subtly wrong is worse than no metric: it makes
a regression look like an improvement, and every Phase 3 decision would be made
on it.
"""

from __future__ import annotations

import math

import pytest

from app.evals.retrieval_metrics import (
    aggregate,
    dcg_at_k,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)

# ── recall@k ────────────────────────────────────────────────

def test_recall_all_relevant_retrieved():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5


def test_recall_respects_the_cutoff():
    """A relevant item below k does not count — that is the whole point."""
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
    assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0


def test_recall_is_zero_when_nothing_is_relevant():
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_recall_ignores_duplicates_in_retrieved():
    assert recall_at_k(["a", "a", "a"], ["a", "b"], k=3) == 0.5


# ── precision@k ─────────────────────────────────────────────

def test_precision_divides_by_k_not_by_result_count():
    """Returning 2 results when 5 were requested is itself a precision cost."""
    assert precision_at_k(["a", "b"], ["a", "b"], k=5) == pytest.approx(0.4)
    assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_precision_zero_when_no_hits():
    assert precision_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_precision_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(["a"], ["a"], k=0)


# ── MRR ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        (["a", "x", "y"], 1.0),
        (["x", "a", "y"], 0.5),
        (["x", "y", "a"], 1 / 3),
        (["x", "y", "z"], 0.0),
    ],
)
def test_reciprocal_rank_uses_the_first_hit(retrieved, expected):
    assert reciprocal_rank(retrieved, ["a"]) == pytest.approx(expected)


def test_reciprocal_rank_takes_the_earliest_of_several_relevant():
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == pytest.approx(0.5)


# ── DCG / nDCG ──────────────────────────────────────────────

def test_dcg_matches_hand_computation():
    # gains 1,0,1 -> 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5
    assert dcg_at_k([1.0, 0.0, 1.0], k=3) == pytest.approx(1.5)


def test_ndcg_is_one_for_perfect_ordering():
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)


def test_ndcg_penalises_a_late_relevant_item():
    """The property recall@k is blind to, and the one a reranker moves."""
    early = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
    late = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
    assert early == 1.0
    assert late < early
    assert late == pytest.approx(1 / math.log2(4))


def test_ndcg_recall_and_ndcg_diverge():
    """Same recall, different nDCG — this is why both are reported."""
    a = ["a", "x", "y"]
    b = ["x", "y", "a"]
    assert recall_at_k(a, ["a"], 3) == recall_at_k(b, ["a"], 3)
    assert ndcg_at_k(a, ["a"], 3) > ndcg_at_k(b, ["a"], 3)


def test_ndcg_zero_when_nothing_relevant_retrieved():
    assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_ndcg_zero_when_no_relevant_set():
    assert ndcg_at_k(["a"], [], k=2) == 0.0


def test_ndcg_with_graded_relevance():
    """A highly-graded item first should beat a weakly-graded item first."""
    graded = {"a": 3.0, "b": 1.0}
    good = ndcg_at_k(["a", "b"], ["a", "b"], k=2, graded=graded)
    bad = ndcg_at_k(["b", "a"], ["a", "b"], k=2, graded=graded)
    assert good == pytest.approx(1.0)
    assert bad < good


def test_ndcg_ideal_is_capped_at_k():
    """With more relevant items than k, the ideal DCG only counts k of them, so
    a perfect top-k still scores 1.0 rather than being penalised."""
    assert ndcg_at_k(["a", "b"], ["a", "b", "c", "d"], k=2) == pytest.approx(1.0)


# ── hit rate ────────────────────────────────────────────────

def test_hit_rate_is_binary():
    assert hit_rate(["x", "a"], ["a"], k=2) == 1.0
    assert hit_rate(["x", "a"], ["a"], k=1) == 0.0


# ── scoring and aggregation ─────────────────────────────────

def test_score_query_reports_every_metric():
    score = score_query(query="q", retrieved=["a", "x"], relevant=["a", "b"], k=2)
    assert score.recall == 0.5
    assert score.precision == 0.5
    assert score.mrr == 1.0
    assert score.hit == 1.0
    assert score.retrieved_count == 2
    assert not score.missed_everything


def test_score_query_flags_a_total_miss():
    score = score_query(query="q", retrieved=["x", "y"], relevant=["a"], k=2)
    assert score.missed_everything


def test_aggregate_is_a_macro_average():
    """Each query counts equally, so a query with many relevant chunks cannot
    swamp several with one each."""
    scores = [
        score_query(query="q1", retrieved=["a"], relevant=["a"], k=1),
        score_query(query="q2", retrieved=["x"], relevant=["b"], k=1),
    ]
    agg = aggregate(scores, k=1)
    assert agg["recall@1"] == pytest.approx(0.5)
    assert agg["queries"] == 2
    assert agg["total_misses"] == 1


def test_aggregate_of_nothing_is_zeroed_not_an_error():
    agg = aggregate([], k=5)
    assert agg["queries"] == 0
    assert agg["recall@5"] == 0.0


def test_aggregate_keys_carry_the_k():
    agg = aggregate([score_query(query="q", retrieved=["a"], relevant=["a"], k=3)], k=3)
    assert "recall@3" in agg
    assert "ndcg@3" in agg
    assert "mrr" in agg
