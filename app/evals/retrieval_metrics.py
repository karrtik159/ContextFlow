"""
Retrieval quality metrics — pure functions over ranked result lists.

These are the metrics Phases 1-3 are actually optimizing and that were, until
now, unmeasurable: there was no corpus, no ranks, and no trace. RAGAS measures
the *generation* half (is the answer faithful to the context it was given); it
cannot tell you whether the right context was retrieved in the first place. A
reranker added without recall@k is cargo-culting, which is why this module
exists before Phase 3 rather than after it.

Everything here takes a ranked list of retrieved ids and a set of relevant ids.
No I/O, no config, no models — so the arithmetic is testable on its own and the
definitions can be checked against the textbook.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant items that appear in the top k.

    The headline retrieval metric: if the right chunk is not in the top k, no
    amount of reranking or synthesis quality can recover it.

    Returns 0.0 when nothing is relevant — a query with no relevant documents
    cannot be scored, and callers should exclude it rather than average it in.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(relevant_set & set(retrieved[:k]))
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    Denominator is k, not len(retrieved[:k]) — returning three results when five
    were asked for is itself a precision cost, and dividing by the shorter list
    would hide it.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return len(set(relevant) & set(retrieved[:k])) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item; 0.0 if none appears.

    Averaged over queries this is MRR. Sensitive to the top of the list in a way
    recall is not — useful when only the first result is really read.
    """
    relevant_set = set(relevant)
    for index, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the log2(rank + 1) discount."""
    return sum(gain / math.log2(index + 1) for index, gain in enumerate(gains[:k], start=1))


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
    *,
    graded: dict[str, float] | None = None,
) -> float:
    """Normalized DCG at k.

    Rewards putting relevant items EARLY, which recall@k is blind to: a relevant
    chunk at position 5 scores the same recall as one at position 1 and a much
    worse nDCG. That difference is exactly what a reranker is supposed to move.

    Args:
        graded: Optional per-item relevance grades. Defaults to binary (1.0 for
            relevant, 0.0 otherwise).
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    def gain(item: str) -> float:
        if graded is not None:
            return graded.get(item, 0.0)
        return 1.0 if item in relevant_set else 0.0

    actual = dcg_at_k([gain(item) for item in retrieved], k)

    # Ideal ordering: every relevant item, best-graded first.
    ideal_gains = sorted((gain(item) for item in relevant_set), reverse=True)
    ideal = dcg_at_k(ideal_gains, k)

    return actual / ideal if ideal > 0 else 0.0


def hit_rate(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant item is in the top k. Coarse, but readable."""
    return 1.0 if set(relevant) & set(retrieved[:k]) else 0.0


@dataclass(frozen=True)
class QueryScore:
    """Per-query metrics. Kept alongside the query so failures are debuggable —
    an aggregate alone tells you the score moved but not which query moved it.
    """

    query: str
    retrieved: list[str]
    relevant: list[str]
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit: float
    retrieved_count: int

    @property
    def missed_everything(self) -> bool:
        return self.hit == 0.0


def score_query(
    *,
    query: str,
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
    graded: dict[str, float] | None = None,
) -> QueryScore:
    relevant_list = list(relevant)
    return QueryScore(
        query=query,
        retrieved=list(retrieved),
        relevant=relevant_list,
        recall=recall_at_k(retrieved, relevant_list, k),
        precision=precision_at_k(retrieved, relevant_list, k),
        mrr=reciprocal_rank(retrieved, relevant_list),
        ndcg=ndcg_at_k(retrieved, relevant_list, k, graded=graded),
        hit=hit_rate(retrieved, relevant_list, k),
        retrieved_count=len(retrieved),
    )


def aggregate(scores: Sequence[QueryScore], k: int) -> dict[str, float]:
    """Mean each metric across queries.

    A plain macro-average: every query counts equally regardless of how many
    relevant chunks it has, so one query with ten relevant chunks cannot swamp
    ten queries with one each.
    """
    if not scores:
        return {
            f"recall@{k}": 0.0,
            f"precision@{k}": 0.0,
            "mrr": 0.0,
            f"ndcg@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            "queries": 0,
            "total_misses": 0,
        }

    n = len(scores)
    return {
        f"recall@{k}": sum(s.recall for s in scores) / n,
        f"precision@{k}": sum(s.precision for s in scores) / n,
        "mrr": sum(s.mrr for s in scores) / n,
        f"ndcg@{k}": sum(s.ndcg for s in scores) / n,
        f"hit_rate@{k}": sum(s.hit for s in scores) / n,
        "queries": n,
        "total_misses": sum(1 for s in scores if s.missed_everything),
    }
