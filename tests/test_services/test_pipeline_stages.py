"""
Pipeline stage wiring — Phase 3.

Runs the real `run_retrieval` with every arm stubbed, so the ordering of stages,
the trace contract, and the session-safety property are all testable without
PostgreSQL, Neo4j, or Mem0.

The `routed_to` contract has its own tests; this file covers the stage contract,
which is the other half of the observability promise: a stage that silently
stops running must fail a test, not merely produce slightly worse answers.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.core.config import settings
from app.services.retrieval import pipeline as pipeline_mod
from app.services.retrieval.contracts import RetrievedChunk

USER_ID = str(uuid.uuid4())


def _chunk(text: str, source: str = "vector", rank: int = 1, score: float = 0.9):
    return RetrievedChunk(
        text=text, source=source, rank=rank, score=score, chunk_id=uuid.uuid4()
    )


class _FakeSavepoint:
    def __init__(self, log):
        self._log = log

    async def __aenter__(self):
        self._log.append(1)

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Minimum AsyncSession surface the pipeline touches: `begin_nested`.

    Every DB arm is wrapped in a savepoint, so the stage tests need a session
    object even though no arm here reaches PostgreSQL.
    """

    def __init__(self):
        self.savepoints: list[int] = []

    def begin_nested(self):
        return _FakeSavepoint(self.savepoints)


@pytest.fixture
def stub_arms(monkeypatch):
    """Replace every arm with a recorder. Returns the call log."""
    calls: list[str] = []

    async def vector(db, **kwargs):
        calls.append("vector")
        return [_chunk("dense hit", "vector", 1, 0.8)]

    async def sparse(db, **kwargs):
        calls.append("bm25")
        return [_chunk("lexical hit", "bm25", 1, 0.5)]

    async def messages(db, **kwargs):
        calls.append("messages")
        return []

    async def graph(**kwargs):
        calls.append("graph")
        return []

    async def memory(**kwargs):
        calls.append("memory")
        return []

    monkeypatch.setattr(pipeline_mod, "search_chunks", vector)
    monkeypatch.setattr(pipeline_mod, "search_chunks_sparse", sparse)
    monkeypatch.setattr(pipeline_mod, "search_messages", messages)
    monkeypatch.setattr(pipeline_mod, "search_graph", graph)
    monkeypatch.setattr(pipeline_mod, "search_memory", memory)
    return calls


@pytest.fixture(autouse=True)
def _no_real_reranker(monkeypatch):
    """Default to rerank disabled; individual tests opt in with a fake."""
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)


async def _run(db=None, **overrides):
    kwargs = {
        "user_id": USER_ID,
        "original_query": "how do I roll back release 20260714-af",
        "retrieval_query": "how do I roll back release 20260714-af",
        "query_embedding": [0.1] * 8,
    }
    kwargs.update(overrides)
    return await pipeline_mod.run_retrieval(db or FakeSession(), **kwargs)


def _stage_names(trace) -> list[str]:
    return [s.name for s in trace.stages]


class TestStageContract:
    async def test_rewrite_stage_is_recorded(self, stub_arms):
        trace = await _run()
        assert "rewrite" in _stage_names(trace)

    async def test_rerank_stage_is_recorded_even_when_disabled(self, stub_arms):
        """A skipped rerank and a rerank that changed nothing must be
        distinguishable in the trace, so the stage records either way."""
        trace = await _run()
        rerank = next(s for s in trace.stages if s.name == "rerank")
        assert rerank.metadata["ran"] is False
        assert rerank.metadata["reason"] == "disabled"

    async def test_sparse_arm_appears_in_the_trace(self, stub_arms):
        trace = await _run()
        assert "retrieve:bm25" in _stage_names(trace)

    async def test_stages_are_in_pipeline_order(self, stub_arms):
        names = _stage_names(await _run())
        assert names.index("rewrite") < names.index("retrieve:fanout")
        assert names.index("retrieve:fanout") < names.index("fuse:rrf")
        assert names.index("fuse:rrf") < names.index("rerank")
        assert names.index("rerank") < names.index("threshold")

    async def test_identifiers_reach_the_sparse_query(self, stub_arms):
        trace = await _run()
        rewrite = next(s for s in trace.stages if s.name == "rewrite")
        assert "20260714-af" in rewrite.metadata["identifiers"]


class TestSparseToggle:
    async def test_sparse_can_be_disabled(self, stub_arms, monkeypatch):
        monkeypatch.setattr(settings, "SPARSE_ENABLED", False)
        trace = await _run()
        assert "bm25" not in stub_arms
        assert "retrieve:bm25" not in _stage_names(trace)

    async def test_sparse_results_are_fused_with_dense(self, stub_arms):
        trace = await _run()
        texts = [c.text for c in trace.final_chunks]
        assert "dense hit" in texts
        assert "lexical hit" in texts


class TestSessionSafety:
    """The Phase 2 fan-out ran the dense and message arms in one
    `asyncio.gather` over a shared AsyncSession. SQLAlchemy rejects that:
    `InvalidRequestError: This session is provisioning a new connection;
    concurrent operations are not permitted`. Phase 3 would have made it three
    concurrent DB arms."""

    async def test_db_arms_never_overlap(self, monkeypatch):
        active = 0
        max_active = 0

        async def db_arm(db, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)  # yield, so an overlap would be observed
            active -= 1
            return []

        async def external(**kwargs):
            await asyncio.sleep(0.01)
            return []

        monkeypatch.setattr(pipeline_mod, "search_chunks", db_arm)
        monkeypatch.setattr(pipeline_mod, "search_chunks_sparse", db_arm)
        monkeypatch.setattr(pipeline_mod, "search_messages", db_arm)
        monkeypatch.setattr(pipeline_mod, "search_graph", external)
        monkeypatch.setattr(pipeline_mod, "search_memory", external)

        await _run()

        assert max_active == 1, (
            f"{max_active} PostgreSQL arms overlapped on one AsyncSession; "
            "SQLAlchemy raises InvalidRequestError on concurrent session use."
        )

    async def test_a_failing_db_arm_does_not_poison_the_next_one(self, monkeypatch):
        """PostgreSQL aborts the whole transaction on any statement error, so
        without a savepoint the first failing arm makes every later arm fail
        with InFailedSQLTransactionError. Because `_run_arm` swallows arm
        failures, that cascade is SILENT: the trace shows three independent
        'arm failed' entries whose stated causes are all the same downstream
        symptom, and the real error survives only in the first.
        """
        session = FakeSession()

        async def failing(db, **kwargs):
            raise RuntimeError("embedding dimension mismatch")

        async def working(db, **kwargs):
            return [_chunk("survived")]

        async def external(**kwargs):
            return []

        monkeypatch.setattr(pipeline_mod, "search_chunks", failing)
        monkeypatch.setattr(pipeline_mod, "search_chunks_sparse", working)
        monkeypatch.setattr(pipeline_mod, "search_messages", working)
        monkeypatch.setattr(pipeline_mod, "search_graph", external)
        monkeypatch.setattr(pipeline_mod, "search_memory", external)

        trace = await _run(db=session, original_query="q", retrieval_query="q")

        # Every DB arm ran inside its own savepoint.
        assert len(session.savepoints) == 3

        vector_stage = next(s for s in trace.stages if s.name == "retrieve:vector")
        assert "embedding dimension mismatch" in vector_stage.metadata["error"]

        # The arms after the failure still returned results rather than
        # inheriting the aborted transaction.
        for name in ("retrieve:bm25", "retrieve:messages"):
            stage = next(s for s in trace.stages if s.name == name)
            assert stage.metadata["error"] is None
            assert stage.metadata["count"] == 1

    async def test_external_arms_still_overlap_the_db_group(self, monkeypatch):
        """Serializing the DB arms must not serialize Neo4j and Mem0 too —
        those are the network-latency arms the concurrency was for."""
        order: list[str] = []

        async def db_arm(db, **kwargs):
            order.append("db_start")
            await asyncio.sleep(0.02)
            order.append("db_end")
            return []

        async def external(**kwargs):
            order.append("ext")
            return []

        monkeypatch.setattr(pipeline_mod, "search_chunks", db_arm)
        monkeypatch.setattr(pipeline_mod, "search_chunks_sparse", db_arm)
        monkeypatch.setattr(pipeline_mod, "search_messages", db_arm)
        monkeypatch.setattr(pipeline_mod, "search_graph", external)
        monkeypatch.setattr(pipeline_mod, "search_memory", external)

        await _run()

        # Both external arms complete before the DB group finishes.
        assert order.index("ext") < order.index("db_end")


class TestRerankIntegration:
    async def test_rerank_reorders_the_final_context(self, stub_arms, monkeypatch):
        monkeypatch.setattr(settings, "RERANK_ENABLED", True)

        async def fake_rerank(query, documents):
            # Score the lexical hit highest, whatever fusion decided.
            return [0.99 if "lexical" in doc else 0.01 for doc in documents]

        monkeypatch.setattr("app.services.reranker.rerank_async", fake_rerank)

        trace = await _run()

        assert trace.final_chunks[0].text == "lexical hit"
        rerank = next(s for s in trace.stages if s.name == "rerank")
        assert rerank.metadata["ran"] is True

    async def test_unavailable_reranker_does_not_fail_the_request(
        self, stub_arms, monkeypatch
    ):
        monkeypatch.setattr(settings, "RERANK_ENABLED", True)

        async def boom(query, documents):
            from app.services.reranker import RerankUnavailable

            raise RerankUnavailable("weights not cached")

        monkeypatch.setattr("app.services.reranker.rerank_async", boom)

        trace = await _run()

        assert trace.has_context
        rerank = next(s for s in trace.stages if s.name == "rerank")
        assert rerank.metadata["ran"] is False

    async def test_citation_labels_follow_the_reranked_order(
        self, stub_arms, monkeypatch
    ):
        """Labels are assigned after reranking, so `[1]` is the chunk the
        cross-encoder ranked first — not the one fusion did."""
        monkeypatch.setattr(settings, "RERANK_ENABLED", True)

        async def fake_rerank(query, documents):
            return [0.99 if "lexical" in doc else 0.01 for doc in documents]

        monkeypatch.setattr("app.services.reranker.rerank_async", fake_rerank)

        trace = await _run()

        assert trace.final_chunks[0].citation_label == "[1]"
        assert trace.final_chunks[0].text == "lexical hit"


class TestFlagGatedProbesCostNothingByDefault:
    async def test_no_extra_embedding_calls_on_the_default_path(
        self, stub_arms, monkeypatch
    ):
        """HyDE and multi-query are off, so the caller's single embedding is
        the only one — the Phase 2 one-LLM-call, one-embedding budget holds."""

        async def explode(text):
            raise AssertionError("default path must not embed anything extra")

        monkeypatch.setattr("app.services.embeddings.embed_text_async_safe", explode)

        trace = await _run()

        assert trace.has_context
