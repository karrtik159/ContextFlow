"""
Reciprocal Rank Fusion — the actual algorithm, in Python.

Until now "Reciprocal Rank Fusion" existed in this repo only as English prose in
two YAML prompts (`support_tasks.yaml`, `support_agents.yaml`), instructing the
model to perform it. Both of those strings are deleted in the same commit as
this file. This is the implementation they described.

RRF scores a document by the sum, over every ranked list it appears in, of
1 / (k + rank). It needs no score calibration between sources, which is what
makes it the right choice here: a cosine similarity, a graph hop count, and a
Mem0 relevance score are not comparable, but their ranks are.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from app.services.retrieval.contracts import RetrievedChunk

# The canonical constant from Cormack et al. (2009). Large relative to typical
# result-set sizes, which damps the difference between rank 1 and rank 2 so a
# single source cannot dominate the fused ordering on its own.
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedChunk:
    """A chunk plus the evidence for its fused position."""

    chunk: RetrievedChunk
    fused_score: float
    # Which source contributed, and at what rank — this is what makes a fused
    # ordering explainable instead of merely plausible.
    contributions: dict[str, int]

    @property
    def source_count(self) -> int:
        return len(self.contributions)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[FusedChunk]:
    """Fuse ranked lists into one ordering.

    Args:
        ranked_lists: One list per source, each already ordered best-first.
            Each item's `rank` is used as given, so a source that ranks its own
            results is respected; ties in the input produce ties in the output.
        k: RRF constant. Higher flattens the contribution curve.
        weights: Optional per-source multipliers, keyed by source name. Absent
            sources default to 1.0. Use to down-weight a noisy arm without
            removing it.

    Returns:
        Fused chunks, best first. Ties break on source_count (an item found by
        two arms outranks one found by a single arm at the same score), then on
        the item's best rank, then on text — so the order is fully deterministic
        and does not depend on dict iteration or input list ordering.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, int]] = {}
    representatives: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for item in ranked:
            if item.rank < 1:
                raise ValueError(
                    f"RRF requires 1-based ranks; got rank={item.rank} from "
                    f"source '{item.source}'. A rank of 0 would over-weight it."
                )
            key = item.dedup_key()
            weight = 1.0 if weights is None else weights.get(item.source, 1.0)
            scores[key] = scores.get(key, 0.0) + weight / (k + item.rank)

            per_source = contributions.setdefault(key, {})
            # If one source returns the same item twice, keep its best rank.
            if item.source not in per_source or item.rank < per_source[item.source]:
                per_source[item.source] = item.rank

            # Keep the representative from the source that ranked it highest,
            # so the surviving text/heading is the best-ranked version of it.
            incumbent = representatives.get(key)
            if incumbent is None or item.rank < incumbent.rank:
                representatives[key] = item

    fused = [
        FusedChunk(
            chunk=representatives[key],
            fused_score=score,
            contributions=dict(sorted(contributions[key].items())),
        )
        for key, score in scores.items()
    ]

    fused.sort(
        key=lambda f: (
            -f.fused_score,
            -f.source_count,
            min(f.contributions.values()),
            f.chunk.text,
        )
    )
    return fused


def assign_citation_labels(fused: Sequence[FusedChunk]) -> list[RetrievedChunk]:
    """Stamp 1-based citation labels in final order.

    Done after fusion because a chunk's label depends on where it finally lands.
    The label is what the synthesis prompt shows and what post-hoc citation
    validation resolves against, so it must be assigned exactly once.
    """
    return [
        replace(item.chunk, citation_label=f"[{index}]", rank=index)
        for index, item in enumerate(fused, start=1)
    ]
