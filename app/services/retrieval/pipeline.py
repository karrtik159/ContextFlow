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
from app.services.retrieval.contracts import RetrievedChunk, RetrievalTrace, StageRecord
from app.services.retrieval.fusion import assign_citation_labels, reciprocal_rank_fusion
from app.services.retrieval.sources import (
    search_chunks,
    search_graph,
    search_memory,
    search_messages,
)

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

    # ── Fan out ─────────────────────────────────────────────
    candidates = settings.RETRIEVAL_CANDIDATES_PER_SOURCE
    with _Timer() as fanout_timer:
        arm_results = await asyncio.gather(
            _run_arm(
                "vector",
                lambda: search_chunks(
                    db,
                    query_embedding=query_embedding,
                    user_id=scoped_uuid,
                    limit=candidates,
                    min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
                ),
            ),
            _run_arm(
                "messages",
                lambda: search_messages(
                    db,
                    query_embedding=query_embedding,
                    user_id=scoped_uuid,
                    limit=settings.RETRIEVAL_MESSAGE_CANDIDATES,
                    min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
                ),
            ),
            _run_arm("graph", lambda: search_graph(query=retrieval_query, user_id=user_id)),
            _run_arm("memory", lambda: search_memory(query=retrieval_query, user_id=user_id)),
        )

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
            input_summary=f"{len(ranked_lists)} arms, concurrent",
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

    # ── Threshold + truncate ────────────────────────────────
    with _Timer() as cut_timer:
        selected = fused[:top_k]
        final = assign_citation_labels(selected)
    trace.final_chunks = final
    trace.record(
        StageRecord(
            name="threshold",
            latency_ms=cut_timer.ms,
            input_summary=f"{len(fused)} fused",
            output_summary=(
                f"{len(final)} selected" if final else "no context cleared the relevance floor"
            ),
            metadata={
                "top_k": top_k,
                "min_similarity": settings.RETRIEVAL_MIN_SIMILARITY,
                "empty": not final,
            },
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
