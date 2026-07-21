"""
The deterministic retrieval pipeline.

Replaces an LLM ReAct loop. Previously `context_gatherer` held three tools with
`max_iter=10` and called them one at a time, each gated on a model turn —
roughly 7-15 sequential LLM round-trips and 3-5 redundant embedding calls per
knowledge query, with a 120s timeout that was a realistic p99 rather than a
safety margin. None of that was a retrieval decision the model was better placed
to make than code.

Here the arms run concurrently, fusion is arithmetic, the relevance floor is a
number, and the only LLM call in the whole request is the final synthesis.

Every stage emits a StageRecord. The trace is the product: it is what makes
retrieval quality measurable rather than anecdotal, and it is what Phase 5's
recall@k and nDCG are computed from.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.retrieval.contracts import RetrievalTrace, RetrievedChunk, StageRecord
from app.services.retrieval.fusion import assign_citation_labels, reciprocal_rank_fusion
from app.services.retrieval.rerank import rerank_fused
from app.services.retrieval.rewrite import rewrite_query
from app.services.retrieval.sources import (
    search_chunks,
    search_chunks_sparse,
    search_graph,
    search_memory,
    search_messages,
)
from app.services.retrieval.sufficiency import assess_sufficiency

logger = logging.getLogger(__name__)

# The corpus is the authority; conversation history and memory personalise but
# should not outvote it. Graph sits between: structural, but sparse and noisy.
SOURCE_WEIGHTS = {
    "vector": 1.0,
    "bm25": 1.0,
    "graph": 0.6,
    "memory": 0.5,
    "messages": 0.4,
}


class _Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self._t0) * 1000)
        return False


async def _run_arm(
    name: str,
    coro_factory: Callable[[], Awaitable[list[RetrievedChunk]]],
) -> tuple[str, list[RetrievedChunk], int, str | None]:
    """Run one arm, timing it and converting failure into an empty result.

    One arm failing must degrade the fused ordering, never fail the request —
    but the failure is recorded in the trace rather than swallowed, so a
    permanently broken arm is visible instead of looking like a quiet corpus.
    """
    t0 = time.perf_counter()
    try:
        items = await coro_factory()
        error = None
    except Exception as exc:
        logger.warning("Retrieval arm '%s' failed: %s", name, exc, exc_info=True)
        items, error = [], f"{type(exc).__name__}: {exc}"
    return name, items, int((time.perf_counter() - t0) * 1000), error


ArmResult = tuple[str, list[RetrievedChunk], int, str | None]


async def _run_db_arm(
    db: AsyncSession,
    name: str,
    coro_factory: Callable[[], Awaitable[list[RetrievedChunk]]],
) -> ArmResult:
    """Run one PostgreSQL arm inside a SAVEPOINT.

    The savepoint is what keeps one arm's failure from becoming every
    subsequent arm's failure. PostgreSQL aborts the whole transaction on any
    statement error, so once `search_chunks` fails — a dimension mismatch is
    the documented live risk (CLAUDE.md: EMBEDDING_DIMENSIONS is load-bearing
    in three places) — every later statement on that session raises
    `InFailedSQLTransactionError` instead of doing its job.

    Because `_run_arm` deliberately swallows arm failures, that cascade is
    silent: the trace records three independent "arm failed" entries whose
    stated causes are all the same downstream symptom, and the real error
    appears only in the first one. Verified against PostgreSQL 16 — a failed
    statement does poison the next, and rolling back to a savepoint restores
    the session while preserving work the request did before retrieval.

    A plain `db.rollback()` would also clear the error, but it would discard
    that earlier work; the caller owns this session and its transaction.
    """
    try:
        async with db.begin_nested():
            return await _run_arm(name, coro_factory)
    except Exception as exc:  # pragma: no cover - savepoint teardown only
        logger.warning("Savepoint teardown failed for arm '%s': %s", name, exc)
        return name, [], 0, f"{type(exc).__name__}: {exc}"


async def _run_db_arm_value(db: AsyncSession, factory):
    """Run a non-arm PostgreSQL call inside a SAVEPOINT.

    Same reasoning as `_run_db_arm`: any statement error aborts the whole
    transaction, and this call happens after retrieval has already succeeded.
    Letting it poison the session would turn an optional check into a failure
    of work that was already done.
    """
    async with db.begin_nested():
        return await factory()


async def _run_db_arms_serially(
    db: AsyncSession,
    arms: list[tuple[str, Callable[[], Awaitable[list[RetrievedChunk]]]]],
) -> list[ArmResult]:
    """Run the PostgreSQL-backed arms one at a time on the shared session.

    A single `AsyncSession` CANNOT be used concurrently. SQLAlchemy detects it
    and raises `InvalidRequestError: This session is provisioning a new
    connection; concurrent operations are not permitted` — verified against the
    installed version, not assumed. The Phase 2 fan-out put the dense corpus arm
    and the message arm in the same `asyncio.gather`, which is that pattern; it
    survived review because the arms are short and the failure needs them to
    actually overlap in the event loop.

    Serializing here rather than opening a session per arm is deliberate. The
    caller owns the session and its transaction — the retrieval arms must read
    the same snapshot the request is already in, and spawning sessions from a
    function that was handed one would silently double the connection use per
    request under the pool the caller sized.

    Little wall-clock is lost: these are index lookups against the same
    connection, and the arms worth overlapping (Neo4j, Mem0) are the network
    ones, which still run concurrently with this whole group.
    """
    return [await _run_db_arm(db, name, factory) for name, factory in arms]


async def run_retrieval(
    db: AsyncSession,
    *,
    user_id: str,
    original_query: str,
    retrieval_query: str,
    query_embedding: list[float],
    top_k: int | None = None,
) -> RetrievalTrace:
    """Retrieve, fuse, and threshold. Makes no LLM calls.

    Args:
        user_id: Resolved tenant scope. Required.
        original_query: What the user actually typed — recorded, not searched.
        retrieval_query: Lightly normalized query text used for the arms.
        query_embedding: Embedding of `retrieval_query`, computed ONCE by the
            caller and reused by every dense arm. The old loop re-embedded per
            tool call.
        top_k: Final context size. Defaults to RETRIEVAL_TOP_K.

    Returns a trace whose `final_chunks` may legitimately be empty.
    """
    if not user_id:
        raise ValueError("run_retrieval requires a user_id — retrieval is tenant-scoped.")

    top_k = top_k or settings.RETRIEVAL_TOP_K
    trace = RetrievalTrace(
        trace_id=uuid.uuid4(),
        user_id=user_id,
        original_query=original_query,
        normalized_query=retrieval_query,
    )
    started = time.perf_counter()

    try:
        scoped_uuid = uuid.UUID(str(user_id))
    except (TypeError, ValueError):
        # Fail closed. An unparseable scope returns nothing; it never widens.
        trace.record(
            StageRecord(
                name="scope",
                latency_ms=0,
                input_summary=f"user_id={user_id!r}",
                output_summary="rejected: unparseable scope",
                metadata={"failed_closed": True},
            )
        )
        trace.total_latency_ms = int((time.perf_counter() - started) * 1000)
        return trace

    # ── Rewrite ─────────────────────────────────────────────
    # Deterministic by default and free. HyDE / multi-query cost an LLM call
    # each and are flag-gated off, so the default path still makes exactly one
    # LLM call per knowledge query (synthesis), as Phase 2 established.
    with _Timer() as rewrite_timer:
        rewritten = await rewrite_query(retrieval_query)
    if rewritten.dense_query != retrieval_query or rewritten.methods:
        trace.rewritten_query = rewritten.dense_query
    trace.record(
        StageRecord(
            name="rewrite",
            latency_ms=rewrite_timer.ms,
            input_summary=retrieval_query[:200],
            output_summary=(
                f"sparse={rewritten.sparse_query[:120]!r} "
                f"identifiers={rewritten.identifiers}"
            ),
            metadata={
                "methods": rewritten.methods,
                "identifiers": rewritten.identifiers,
                "variants": rewritten.variants,
                "hyde": bool(rewritten.hyde_document),
            },
        )
    )

    # Extra dense probes (HyDE / multi-query) need their own embeddings. The
    # caller's embedding is always reused for the primary query — this only
    # embeds text the caller could not have known about.
    extra_embeddings: list[tuple[str, list[float]]] = []
    extra_texts = rewritten.dense_queries[1:]
    if extra_texts:
        from app.services.embeddings import embed_text_async_safe

        for index, text in enumerate(extra_texts):
            vector = await embed_text_async_safe(text)
            if vector:
                extra_embeddings.append((f"vector:rewrite{index + 1}", vector))

    # ── Fan out ─────────────────────────────────────────────
    candidates = settings.RETRIEVAL_CANDIDATES_PER_SOURCE
    with _Timer() as fanout_timer:
        # PostgreSQL arms share the caller's session and therefore run in
        # sequence — see _run_db_arms_serially for why concurrency here is not
        # merely slower but an error.
        db_arms: list[tuple[str, Callable[[], Awaitable[list[RetrievedChunk]]]]] = [
            (
                "vector",
                lambda: search_chunks(
                    db,
                    query_embedding=query_embedding,
                    user_id=scoped_uuid,
                    limit=candidates,
                    min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
                ),
            ),
            (
                "messages",
                lambda: search_messages(
                    db,
                    query_embedding=query_embedding,
                    user_id=scoped_uuid,
                    limit=settings.RETRIEVAL_MESSAGE_CANDIDATES,
                    min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
                ),
            ),
        ]

        if settings.SPARSE_ENABLED:
            db_arms.append(
                (
                    "bm25",
                    lambda: search_chunks_sparse(
                        db,
                        query=rewritten.sparse_query,
                        user_id=scoped_uuid,
                        limit=settings.SPARSE_CANDIDATES,
                        min_rank=settings.SPARSE_MIN_RANK,
                        ts_config=settings.SPARSE_TS_CONFIG,
                    ),
                )
            )

        # Each extra dense probe is its own ranked list, so a chunk that all
        # three rewrites agree on gains RRF weight from that agreement. That is
        # the point of multi-query; it is also why these are flag-gated —
        # unearned agreement is just the vector arm voting three times.
        for arm_name, vector in extra_embeddings:
            db_arms.append(
                (
                    arm_name,
                    lambda v=vector: search_chunks(
                        db,
                        query_embedding=v,
                        user_id=scoped_uuid,
                        limit=candidates,
                        min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
                    ),
                )
            )

        # The external arms hold no PostgreSQL session, so they overlap freely
        # with the whole DB group — which is where the concurrency was actually
        # worth having, since Neo4j and Mem0 are the network-latency arms.
        db_results, *external_results = await asyncio.gather(
            _run_db_arms_serially(db, db_arms),
            _run_arm("graph", lambda: search_graph(query=retrieval_query, user_id=user_id)),
            _run_arm("memory", lambda: search_memory(query=retrieval_query, user_id=user_id)),
        )
        arm_results = [*db_results, *external_results]

    ranked_lists: list[list[RetrievedChunk]] = []
    for name, items, latency_ms, error in arm_results:
        ranked_lists.append(items)
        trace.record(
            StageRecord(
                name=f"retrieve:{name}",
                latency_ms=latency_ms,
                input_summary=retrieval_query[:200],
                output_summary=f"{len(items)} results",
                metadata={
                    "count": len(items),
                    "top_score": items[0].score if items else None,
                    "error": error,
                },
            )
        )

    trace.record(
        StageRecord(
            name="retrieve:fanout",
            latency_ms=fanout_timer.ms,
            input_summary=f"{len(ranked_lists)} arms (db serial, external concurrent)",
            output_summary=f"{sum(len(r) for r in ranked_lists)} total candidates",
            metadata={
                "arms": [name for name, *_ in arm_results],
                "min_similarity": settings.RETRIEVAL_MIN_SIMILARITY,
                "candidates_per_source": candidates,
            },
        )
    )

    # ── Fuse ────────────────────────────────────────────────
    with _Timer() as fuse_timer:
        fused = reciprocal_rank_fusion(
            ranked_lists, k=settings.RRF_K, weights=SOURCE_WEIGHTS
        )
    trace.record(
        StageRecord(
            name="fuse:rrf",
            latency_ms=fuse_timer.ms,
            input_summary=f"{sum(len(r) for r in ranked_lists)} candidates from {len(ranked_lists)} arms",
            output_summary=f"{len(fused)} unique after dedup",
            metadata={
                "k": settings.RRF_K,
                "weights": SOURCE_WEIGHTS,
                "multi_source_hits": sum(1 for f in fused if f.source_count > 1),
                "top": [
                    {"score": round(f.fused_score, 6), "sources": f.contributions}
                    for f in fused[:5]
                ],
            },
        )
    )

    # ── Rerank ──────────────────────────────────────────────
    # The stage that makes "retrieve wide" pay off. Fusion orders by rank
    # agreement between arms, which is a proxy; the cross-encoder reads the
    # query and the chunk together and scores relevance directly.
    # The reason is set from the actual condition rather than defaulted, so the
    # persisted trace distinguishes "reranking is switched off" from "there was
    # nothing to rerank" — two very different things to read in an incident.
    rerank_meta: dict = {
        "ran": False,
        "reason": "disabled" if not settings.RERANK_ENABLED else "no candidates",
    }
    ordered = fused
    with _Timer() as rerank_timer:
        if settings.RERANK_ENABLED and fused:
            ordered, rerank_meta = await rerank_fused(
                retrieval_query,
                fused,
                candidates=settings.RERANK_CANDIDATES,
                min_score=settings.RERANK_MIN_SCORE,
            )
    trace.record(
        StageRecord(
            name="rerank",
            latency_ms=rerank_timer.ms,
            input_summary=f"{len(fused)} fused, top {settings.RERANK_CANDIDATES} scored",
            output_summary=(
                f"{rerank_meta.get('positions_changed', 0)} positions changed"
                if rerank_meta.get("ran")
                else f"skipped: {rerank_meta.get('reason')}"
            ),
            metadata={"model": settings.RERANK_MODEL, **rerank_meta},
        )
    )

    # ── Threshold + truncate ────────────────────────────────
    with _Timer() as cut_timer:
        selected = ordered[:top_k]
        final = assign_citation_labels(selected)
    trace.final_chunks = final
    trace.record(
        StageRecord(
            name="threshold",
            latency_ms=cut_timer.ms,
            input_summary=f"{len(ordered)} candidates after rerank",
            output_summary=(
                f"{len(final)} selected" if final else "no context cleared the relevance floor"
            ),
            metadata={
                "top_k": top_k,
                "min_similarity": settings.RETRIEVAL_MIN_SIMILARITY,
                "rerank_min_score": settings.RERANK_MIN_SCORE,
                "reranked": rerank_meta.get("ran", False),
                "empty": not final,
            },
        )
    )

    # ── Sufficiency ─────────────────────────────────────────
    # Relevance says "this context is about the question". Sufficiency says
    # "this context can answer it". Phase 3 measured that the first does not
    # imply the second and that no threshold on a relevance score recovers it,
    # so this is a separate stage with its own signal rather than another dial.
    #
    # It runs LAST, on the final context, because that is what the synthesis
    # model will actually see — gating on the pre-truncation candidate pool
    # would approve context the answer is not built from.
    with _Timer() as sufficiency_timer:
        if settings.SUFFICIENCY_ENABLED:
            # Savepoint for the same reason the arms have one: this runs after
            # retrieval already succeeded, and an error here must not poison a
            # transaction that has real work in it.
            verdict = await _run_db_arm_value(
                db,
                lambda: assess_sufficiency(
                    db, retrieval_query, final, user_id=scoped_uuid
                ),
            )
        else:
            verdict = await assess_sufficiency(
                db, retrieval_query, final, user_id=scoped_uuid
            )
    if not verdict.sufficient:
        # The honest-empty path. Dropping the chunks rather than flagging them
        # keeps `has_context` the single place callers ask this question.
        trace.final_chunks = []
    trace.record(
        StageRecord(
            name="sufficiency",
            latency_ms=sufficiency_timer.ms,
            input_summary=f"{len(final)} chunks, {retrieval_query[:120]!r}",
            output_summary=(
                "sufficient" if verdict.sufficient else f"abstain: {verdict.reason}"
            ),
            metadata=verdict.to_metadata(),
        )
    )

    trace.total_latency_ms = int((time.perf_counter() - started) * 1000)
    return trace


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as fenced, numbered, citable blocks.

    Retrieved text is DATA, not instructions (§5.4). It is fenced and explicitly
    labelled untrusted so that a stored-injection payload reads as quoted
    material rather than as a directive the model should follow.
    """
    if not chunks:
        return "NO CONTEXT RETRIEVED."

    parts: list[str] = []
    for chunk in chunks:
        header = chunk.citation_label
        if chunk.heading_path:
            header = f"{header} {chunk.heading_path}"
        parts.append(
            f"<<<CONTEXT {header} (source: {chunk.source})>>>\n"
            f"{chunk.text.strip()}\n"
            f"<<<END CONTEXT {chunk.citation_label}>>>"
        )
    return "\n\n".join(parts)
