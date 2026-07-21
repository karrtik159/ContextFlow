"""
The rerank stage: reorder fused candidates by direct query-document relevance.

Retrieve wide, rerank narrow. Before this stage existed, `RETRIEVAL_TOP_K = 5`
was both the retrieval width AND the final context, so a chunk that fusion
placed 6th was unrecoverable — no later stage ever saw it. Now the arms fan out
to `RETRIEVAL_CANDIDATES_PER_SOURCE`, fusion orders them, and the cross-encoder
gets the top `RERANK_CANDIDATES` before anything is truncated to `top_k`.

Failure here is always a degradation, never an error. If the model will not
load, the fused ordering is already a usable answer; returning a 500 because
the *optional quality improvement* is unavailable would be strictly worse than
the pre-Phase-3 behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from app.services.retrieval.fusion import FusedChunk

logger = logging.getLogger(__name__)


async def rerank_fused(
    query: str,
    fused: list[FusedChunk],
    *,
    candidates: int,
    min_score: float = 0.0,
) -> tuple[list[FusedChunk], dict[str, Any]]:
    """Rescore the top `candidates` with a cross-encoder.

    Returns `(ordering, metadata)`. `metadata` always describes what happened —
    including "it did not run and why" — because a silently-skipped rerank looks
    identical in the response to a rerank that ran and changed nothing.

    Only the head of the list is rescored. The tail keeps its fused order and is
    appended after every reranked item, so a candidate the cross-encoder never
    saw can never outrank one it scored.
    """
    from app.services.reranker import RerankUnavailable, rerank_async

    if not fused:
        return fused, {"ran": False, "reason": "no candidates"}

    head = fused[:candidates]
    tail = fused[candidates:]

    try:
        scores = await rerank_async(query, [_rerank_text(f) for f in head])
        # `strict=True` is load-bearing. A model that returns fewer scores than
        # documents would otherwise pair each score with the WRONG chunk and
        # produce a confidently mis-ordered result with no error anywhere —
        # the worst failure mode this stage has. Raising here degrades to the
        # fused order instead, which is merely unimproved.
        scored = _apply_scores(head, scores)
    except RerankUnavailable as exc:
        logger.warning("Rerank unavailable, keeping fused order: %s", exc)
        return fused, {"ran": False, "reason": f"unavailable: {exc}"}
    except Exception as exc:
        logger.warning("Rerank failed, keeping fused order: %s", exc, exc_info=True)
        return fused, {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}

    kept = [item for item in scored if item.chunk.score >= min_score]
    dropped = len(scored) - len(kept)

    # Ties break on the original fused position, so the ordering stays
    # deterministic when the cross-encoder is indifferent between two chunks.
    kept.sort(key=lambda f: (-f.chunk.score, f.chunk.metadata.get("fused_rank", 0)))

    moved = sum(
        1
        for new_index, item in enumerate(kept, start=1)
        if item.chunk.metadata.get("fused_rank") != new_index
    )

    return kept + tail, {
        "ran": True,
        "scored": len(scored),
        "kept": len(kept),
        "dropped_below_min_score": dropped,
        "min_score": min_score,
        "positions_changed": moved,
        "top_scores": [round(item.chunk.score, 4) for item in kept[:5]],
    }


def _rerank_text(item: FusedChunk) -> str:
    """What the cross-encoder actually reads for one candidate.

    The heading path is PREPENDED, and that is not cosmetic. A chunk is a slice
    of a document, and the slice frequently does not restate its own subject:
    "Synchronisation runs every four hours, and an account owner may trigger it
    manually once per hour" never says the word "directory". Scored bare
    against "How often are encryption keys rotated?", a cross-encoder reads it
    as a strong answer to "how often …" and ranks it above the chunk that
    actually discusses key rotation. Measured on the golden corpus, that exact
    confusion cost recall@3 — reranking scored WORSE than fusion alone until
    the heading path was included.

    `build_context_block` already shows the heading path to the synthesis model
    for the same reason. Scoring a chunk on less context than the model that
    reads it will see is a straightforward mismatch.
    """
    if item.chunk.heading_path:
        return f"{item.chunk.heading_path}\n{item.chunk.text.strip()}"
    return item.chunk.text.strip()


def _apply_scores(head: list[FusedChunk], scores: list[float]) -> list[FusedChunk]:
    """Attach cross-encoder scores to candidates, preserving the fused position."""
    return [
        replace(
            item,
            chunk=replace(
                item.chunk,
                score=score,
                metadata={
                    **item.chunk.metadata,
                    "rerank_score": score,
                    # The pre-rerank position, kept so the trace can show what
                    # the reranker actually changed rather than just its output.
                    "fused_rank": index + 1,
                    "fused_score": item.fused_score,
                },
            ),
        )
        for index, (item, score) in enumerate(zip(head, scores, strict=True))
    ]
