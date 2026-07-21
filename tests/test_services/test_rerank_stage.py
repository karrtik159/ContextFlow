"""
The rerank stage — Phase 3.

This is where "retrieve wide, rerank narrow" is enforced. Before it existed,
RETRIEVAL_TOP_K was both the retrieval width and the final context, so a chunk
fusion placed 6th could never be recovered. These tests assert that the stage
can promote such a chunk, that it never fails a request when the model is
unavailable, and that its ordering is deterministic.
"""

from __future__ import annotations

import uuid

import pytest

from app.services.retrieval.contracts import RetrievedChunk
from app.services.retrieval.fusion import FusedChunk
from app.services.retrieval.rerank import rerank_fused


def _fused(text: str, fused_score: float, source: str = "vector") -> FusedChunk:
    return FusedChunk(
        chunk=RetrievedChunk(
            text=text,
            source=source,
            rank=1,
            score=fused_score,
            chunk_id=uuid.uuid4(),
        ),
        fused_score=fused_score,
        contributions={source: 1},
    )


def _patch_scores(monkeypatch, scores_by_text: dict[str, float]):
    async def fake_rerank_async(query, documents):
        return [scores_by_text[doc] for doc in documents]

    monkeypatch.setattr("app.services.reranker.rerank_async", fake_rerank_async)


class TestPromotion:
    async def test_a_chunk_below_top_k_can_be_promoted_to_first(self, monkeypatch):
        """The whole point of the stage. Without it, 'sixth' means 'invisible'."""
        fused = [_fused(f"doc{i}", 1.0 - i * 0.1) for i in range(8)]
        _patch_scores(
            monkeypatch,
            {f"doc{i}": (0.99 if i == 5 else 0.1) for i in range(8)},
        )

        ordered, meta = await rerank_fused("q", fused, candidates=8)

        assert meta["ran"] is True
        assert ordered[0].chunk.text == "doc5"
        assert meta["positions_changed"] > 0

    async def test_rerank_score_replaces_the_chunk_score(self, monkeypatch):
        fused = [_fused("a", 0.9), _fused("b", 0.8)]
        _patch_scores(monkeypatch, {"a": 0.2, "b": 0.7})

        ordered, _ = await rerank_fused("q", fused, candidates=2)

        assert ordered[0].chunk.text == "b"
        assert ordered[0].chunk.score == 0.7

    async def test_pre_rerank_position_is_preserved_in_metadata(self, monkeypatch):
        """The trace must be able to show what the reranker CHANGED, not just
        what it produced — otherwise a no-op rerank and a working one look the
        same in the record."""
        fused = [_fused("a", 0.9), _fused("b", 0.8)]
        _patch_scores(monkeypatch, {"a": 0.2, "b": 0.7})

        ordered, _ = await rerank_fused("q", fused, candidates=2)

        assert ordered[0].chunk.metadata["fused_rank"] == 2
        assert ordered[0].chunk.metadata["rerank_score"] == 0.7
        assert ordered[0].chunk.metadata["fused_score"] == 0.8


class TestCandidateWindow:
    async def test_only_the_head_is_scored(self, monkeypatch):
        seen = {}

        async def fake_rerank_async(query, documents):
            seen["n"] = len(documents)
            return [0.5] * len(documents)

        monkeypatch.setattr("app.services.reranker.rerank_async", fake_rerank_async)
        fused = [_fused(f"doc{i}", 1.0) for i in range(30)]

        ordered, meta = await rerank_fused("q", fused, candidates=10)

        assert seen["n"] == 10
        assert meta["scored"] == 10
        assert len(ordered) == 30

    async def test_unscored_tail_never_outranks_a_scored_chunk(self, monkeypatch):
        """A candidate the cross-encoder never saw has no score to compare, so
        it must stay behind everything that was scored — including chunks the
        reranker scored badly."""
        fused = [_fused(f"doc{i}", 1.0) for i in range(5)]
        _patch_scores(monkeypatch, {f"doc{i}": 0.01 for i in range(2)})

        ordered, _ = await rerank_fused("q", fused, candidates=2)

        assert [c.chunk.text for c in ordered[:2]] == ["doc0", "doc1"]
        assert [c.chunk.text for c in ordered[2:]] == ["doc2", "doc3", "doc4"]


class TestMinScore:
    async def test_chunks_below_min_score_are_dropped(self, monkeypatch):
        fused = [_fused("a", 0.9), _fused("b", 0.8), _fused("c", 0.7)]
        _patch_scores(monkeypatch, {"a": 0.9, "b": 0.05, "c": 0.6})

        ordered, meta = await rerank_fused("q", fused, candidates=3, min_score=0.5)

        assert [c.chunk.text for c in ordered] == ["a", "c"]
        assert meta["dropped_below_min_score"] == 1

    async def test_min_score_zero_drops_nothing(self, monkeypatch):
        fused = [_fused("a", 0.9), _fused("b", 0.8)]
        _patch_scores(monkeypatch, {"a": 0.0, "b": 0.0})

        ordered, meta = await rerank_fused("q", fused, candidates=2, min_score=0.0)

        assert len(ordered) == 2
        assert meta["dropped_below_min_score"] == 0

    async def test_everything_below_floor_yields_the_honest_empty_path(self, monkeypatch):
        fused = [_fused("a", 0.9), _fused("b", 0.8)]
        _patch_scores(monkeypatch, {"a": 0.1, "b": 0.1})

        ordered, meta = await rerank_fused("q", fused, candidates=2, min_score=0.9)

        assert ordered == []
        assert meta["kept"] == 0


class TestDegradation:
    async def test_unavailable_model_keeps_fused_order(self, monkeypatch):
        from app.services.reranker import RerankUnavailable

        async def boom(query, documents):
            raise RerankUnavailable("weights not cached")

        monkeypatch.setattr("app.services.reranker.rerank_async", boom)
        fused = [_fused("a", 0.9), _fused("b", 0.8)]

        ordered, meta = await rerank_fused("q", fused, candidates=2)

        assert [c.chunk.text for c in ordered] == ["a", "b"]
        assert meta["ran"] is False
        assert "unavailable" in meta["reason"]

    async def test_unexpected_error_also_degrades(self, monkeypatch):
        async def boom(query, documents):
            raise RuntimeError("cuda oom")

        monkeypatch.setattr("app.services.reranker.rerank_async", boom)
        fused = [_fused("a", 0.9)]

        ordered, meta = await rerank_fused("q", fused, candidates=1)

        assert len(ordered) == 1
        assert meta["ran"] is False
        assert "RuntimeError" in meta["reason"]

    async def test_timeout_degrades_rather_than_failing(self, monkeypatch):
        async def boom(query, documents):
            raise TimeoutError()

        monkeypatch.setattr("app.services.reranker.rerank_async", boom)
        ordered, meta = await rerank_fused("q", [_fused("a", 0.9)], candidates=1)

        assert len(ordered) == 1
        assert meta["ran"] is False

    async def test_empty_input_is_not_an_error(self):
        ordered, meta = await rerank_fused("q", [], candidates=5)
        assert ordered == []
        assert meta["ran"] is False


class TestDeterminism:
    async def test_ties_break_on_original_fused_position(self, monkeypatch):
        """An indifferent cross-encoder must not shuffle the ordering, or two
        identical requests return different context."""
        fused = [_fused(f"doc{i}", 1.0 - i * 0.01) for i in range(6)]
        _patch_scores(monkeypatch, {f"doc{i}": 0.5 for i in range(6)})

        first, _ = await rerank_fused("q", fused, candidates=6)
        second, _ = await rerank_fused("q", fused, candidates=6)

        assert [c.chunk.text for c in first] == [f"doc{i}" for i in range(6)]
        assert [c.chunk.text for c in first] == [c.chunk.text for c in second]


class TestScoreCountMismatch:
    async def test_short_score_list_is_a_loud_failure(self, monkeypatch):
        """A model returning fewer scores than documents would silently pair
        each score with the wrong chunk under zip()'s default behaviour."""

        async def short(query, documents):
            return [0.5] * (len(documents) - 1)

        monkeypatch.setattr("app.services.reranker.rerank_async", short)
        fused = [_fused("a", 0.9), _fused("b", 0.8)]

        ordered, meta = await rerank_fused("q", fused, candidates=2)

        # It degrades rather than mis-pairing: strict=True raises, and the
        # stage's catch-all turns that into "kept the fused order".
        assert meta["ran"] is False
        assert [c.chunk.text for c in ordered] == ["a", "b"]


@pytest.mark.parametrize("candidates", [1, 2, 5, 100])
async def test_candidate_window_sizes_preserve_every_chunk(monkeypatch, candidates):
    fused = [_fused(f"doc{i}", 1.0) for i in range(5)]
    _patch_scores(monkeypatch, {f"doc{i}": 0.5 for i in range(5)})

    ordered, _ = await rerank_fused("q", fused, candidates=candidates)

    assert len(ordered) == 5
