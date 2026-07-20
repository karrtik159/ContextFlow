"""
Unit tests for Reciprocal Rank Fusion.

Pure arithmetic, no I/O. These exist because RRF previously existed only as an
English instruction in a YAML prompt — the model was asked to run a ranking
algorithm over data that carried no ranks and no scores, so its output could
not be checked. These assertions are the difference.
"""

from __future__ import annotations

import uuid

import pytest

from app.services.retrieval.contracts import RetrievedChunk
from app.services.retrieval.fusion import (
    DEFAULT_RRF_K,
    assign_citation_labels,
    reciprocal_rank_fusion,
)


def chunk(text: str, source: str, rank: int, score: float = 0.5, chunk_id=None) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source=source, rank=rank, score=score, chunk_id=chunk_id
    )


def ranked(source: str, *texts: str) -> list[RetrievedChunk]:
    return [chunk(t, source, i) for i, t in enumerate(texts, start=1)]


# ── Core arithmetic ─────────────────────────────────────────

def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([ranked("vector", "a", "b", "c")])
    assert [f.chunk.text for f in fused] == ["a", "b", "c"]


def test_score_matches_the_rrf_formula():
    fused = reciprocal_rank_fusion([ranked("vector", "a", "b")], k=60)
    assert fused[0].fused_score == pytest.approx(1 / 61)
    assert fused[1].fused_score == pytest.approx(1 / 62)


def test_agreement_across_sources_outranks_a_single_first_place():
    """An item ranked 2nd by two sources beats one ranked 1st by a single
    source — this is the entire point of fusion."""
    a = reciprocal_rank_fusion(
        [
            ranked("vector", "solo", "agreed"),
            ranked("graph", "other", "agreed"),
        ]
    )
    top = a[0]
    assert top.chunk.text == "agreed"
    assert top.fused_score == pytest.approx(2 * (1 / 62))
    assert top.source_count == 2


def test_k_damps_the_rank_one_advantage():
    """Larger k flattens the contribution curve, so the gap between rank 1 and
    the tail narrows and one source dominates the ordering less.

    The lists must agree on order; two mirrored lists give every item the same
    score for any k, which would make this assertion vacuous.
    """
    lists = [ranked("vector", "a", "b", "c"), ranked("graph", "a", "b", "c")]
    small = reciprocal_rank_fusion(lists, k=1)
    large = reciprocal_rank_fusion(lists, k=1000)
    spread_small = small[0].fused_score - small[-1].fused_score
    spread_large = large[0].fused_score - large[-1].fused_score
    assert spread_small > 0, "fixture must produce a non-degenerate spread"
    assert spread_large < spread_small


def test_default_k_is_the_published_constant():
    assert DEFAULT_RRF_K == 60


# ── Deduplication ───────────────────────────────────────────

def test_same_chunk_id_from_two_sources_fuses_into_one_result():
    cid = uuid.uuid4()
    fused = reciprocal_rank_fusion(
        [
            [chunk("body text", "vector", 1, chunk_id=cid)],
            [chunk("body text", "bm25", 1, chunk_id=cid)],
        ]
    )
    assert len(fused) == 1
    assert fused[0].source_count == 2
    assert fused[0].fused_score == pytest.approx(2 / 61)


def test_identical_text_without_ids_still_dedups_within_a_source():
    fused = reciprocal_rank_fusion(
        [[chunk("same", "graph", 1), chunk("same", "graph", 2)]]
    )
    assert len(fused) == 1
    # Both occurrences contribute, but the best rank is what is reported.
    assert fused[0].contributions == {"graph": 1}


def test_same_text_from_different_sources_fuses_as_agreement():
    """Identity is source-independent, so the same fact surfaced by two arms
    counts as agreement.

    Qualifying id-less items by source would score the graph and memory arms in
    isolation and reduce fusion to a weighted concatenation.
    """
    fused = reciprocal_rank_fusion(
        [[chunk("Alice knows Bob", "graph", 1)], [chunk("Alice knows Bob", "memory", 1)]]
    )
    assert len(fused) == 1
    assert fused[0].source_count == 2


def test_dedup_normalizes_whitespace_and_case():
    fused = reciprocal_rank_fusion(
        [[chunk("Alice  knows   Bob", "graph", 1)], [chunk("alice knows bob", "memory", 1)]]
    )
    assert len(fused) == 1


# ── Weights ─────────────────────────────────────────────────

def test_weights_scale_a_source_contribution():
    fused = reciprocal_rank_fusion(
        [ranked("vector", "v"), ranked("memory", "m")],
        weights={"vector": 1.0, "memory": 0.5},
    )
    assert fused[0].chunk.text == "v"
    assert fused[1].fused_score == pytest.approx(0.5 / 61)


def test_zero_weight_keeps_the_item_but_removes_its_influence():
    fused = reciprocal_rank_fusion(
        [ranked("memory", "m")], weights={"memory": 0.0}
    )
    assert len(fused) == 1
    assert fused[0].fused_score == 0.0


def test_absent_source_defaults_to_weight_one():
    fused = reciprocal_rank_fusion([ranked("vector", "v")], weights={"graph": 0.1})
    assert fused[0].fused_score == pytest.approx(1 / 61)


# ── Validation and edge cases ───────────────────────────────

def test_empty_input_yields_empty_output():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], [], []]) == []


def test_zero_rank_is_rejected():
    """A 0 rank would give 1/k, over-weighting it against a legitimate rank 1."""
    with pytest.raises(ValueError, match="1-based"):
        reciprocal_rank_fusion([[chunk("a", "vector", 0)]])


def test_negative_k_is_rejected():
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([ranked("vector", "a")], k=0)


def test_ordering_is_deterministic_regardless_of_input_list_order():
    l1, l2, l3 = ranked("vector", "a", "b"), ranked("graph", "b", "c"), ranked("memory", "c", "a")
    first = [f.chunk.text for f in reciprocal_rank_fusion([l1, l2, l3])]
    second = [f.chunk.text for f in reciprocal_rank_fusion([l3, l2, l1])]
    assert first == second


def test_contributions_report_which_source_ranked_what():
    fused = reciprocal_rank_fusion(
        [ranked("vector", "x", "y"), ranked("graph", "y", "x")]
    )
    for item in fused:
        assert set(item.contributions) == {"vector", "graph"}
        assert all(r >= 1 for r in item.contributions.values())


# ── Citation labels ─────────────────────────────────────────

def test_citation_labels_are_assigned_in_final_order():
    fused = reciprocal_rank_fusion([ranked("vector", "a", "b", "c")])
    labelled = assign_citation_labels(fused)
    assert [c.citation_label for c in labelled] == ["[1]", "[2]", "[3]"]
    assert [c.rank for c in labelled] == [1, 2, 3]


def test_citation_labels_do_not_mutate_the_source_chunks():
    original = ranked("vector", "a")
    assign_citation_labels(reciprocal_rank_fusion([original]))
    assert original[0].citation_label == "", "input chunks must not be mutated"


def test_labels_are_unique():
    fused = reciprocal_rank_fusion(
        [ranked("vector", "a", "b"), ranked("graph", "c", "d")]
    )
    labels = [c.citation_label for c in assign_citation_labels(fused)]
    assert len(labels) == len(set(labels))
