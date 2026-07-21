"""
Cross-encoder reranking — the single source of truth for query-document scoring.

A bi-encoder (the embedding model) turns the query and the document into
vectors *independently* and compares them. It never sees the pair. That is what
makes it fast enough to search a corpus, and it is also why Phase 5 measured
this result: no cosine floor both preserved recall and rejected the
topically-near-but-unanswerable query ("configure SAML SSO", absent from the
corpus, retrieved three chunks at similarity 0.467). Embedding proximity is
not relevance, and no threshold on it can be made to mean relevance.

A cross-encoder reads the query and the document *together* in one forward pass
and emits a relevance score directly. It cannot be used to search a corpus —
it is O(candidates) model calls — which is precisely why it belongs here,
after retrieval has narrowed the field to a few dozen.

The model is loaded lazily behind double-checked locking, exactly like the
SentenceTransformer encoder and the Neo4j driver. Importing this module must
not load weights: `test_reranker_import_does_not_load_model` enforces it, for
the same reason `test_graph_search_import_does_not_connect` exists — unit tests
must run without a model cache, and a 400 MB import is an import-time landmine.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from typing import TYPE_CHECKING, Any

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_cross_encoder: CrossEncoder | None = None
_cross_encoder_lock = threading.Lock()

# Set when loading has already failed, so a request path that cannot load the
# model does not re-attempt (and re-pay the timeout) on every query. Cleared
# only by a process restart — a model that failed to load will not spontaneously
# start loading.
_load_failed: str | None = None


class RerankUnavailable(RuntimeError):
    """The reranker could not run. Callers degrade; they do not fail."""


def _get_cross_encoder() -> CrossEncoder:
    """Thread-safe lazy loader for the cross-encoder.

    Same double-checked-locking shape as `_get_hf_encoder` in
    `app/services/embeddings.py`. The lock matters for the same reason: the
    underlying tokenizer is not safe to initialize concurrently, and CrewAI /
    `asyncio.to_thread` mean several threads can arrive here at once.
    """
    global _cross_encoder, _load_failed

    if _cross_encoder is not None:
        return _cross_encoder
    if _load_failed is not None:
        raise RerankUnavailable(_load_failed)

    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        if _load_failed is not None:
            raise RerankUnavailable(_load_failed)

        try:
            import torch
            from sentence_transformers import CrossEncoder as _CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = settings.RERANK_MODEL
            logger.info("Loading cross-encoder '%s' on device '%s'...", model_name, device)
            _cross_encoder = _CrossEncoder(model_name, device=device)
            logger.info("Cross-encoder ready — model=%s device=%s", model_name, device)
        except Exception as exc:
            _load_failed = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Cross-encoder '%s' failed to load; reranking disabled for this "
                "process and retrieval will fall back to fused order: %s",
                settings.RERANK_MODEL,
                exc,
            )
            raise RerankUnavailable(_load_failed) from exc

    return _cross_encoder


def reset_reranker_for_tests() -> None:
    """Drop the cached model and the failure latch. Tests only."""
    global _cross_encoder, _load_failed
    with _cross_encoder_lock:
        _cross_encoder = None
        _load_failed = None


def _sigmoid(raw: float) -> float:
    if raw < -60:  # exp overflow guard; sigmoid is 0 well before float precision
        return 0.0
    if raw > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-raw))


def _model_applies_activation(encoder: CrossEncoder) -> bool:
    """Whether `predict()` already squashes its output into [0, 1].

    sentence-transformers sets a `Sigmoid()` activation for a single-label head
    (verified: v5.2.3 reports `activation_fn: Sigmoid()` for
    `BAAI/bge-reranker-base`), but the attribute has been renamed across major
    versions, so both spellings are checked and an unrecognised model is
    assumed to emit raw logits.

    This is decided ONCE PER MODEL rather than per score. Deciding per score —
    "pass through if it is already in [0, 1], otherwise sigmoid it" — is not
    monotonic across the boundary: a logit of 0.9 would map to 0.9 while a
    LARGER logit of 1.5 would map to sigmoid(1.5) = 0.82, silently inverting
    the ranking of exactly the two candidates a reranker is meant to separate.
    """
    activation = getattr(encoder, "activation_fn", None)
    if activation is None:
        activation = getattr(encoder, "default_activation_function", None)
    if activation is None:
        return False
    return type(activation).__name__.lower() in {"sigmoid", "softmax"}


def rerank_sync(query: str, documents: list[str]) -> list[float]:
    """Score each document against the query. Blocking; call from a thread.

    Returns one probability per document, in input order. Raises
    `RerankUnavailable` if the model cannot be loaded — the caller decides
    whether that degrades or fails, and in this codebase it always degrades.
    """
    if not documents:
        return []

    encoder = _get_cross_encoder()
    pairs = [(query, doc) for doc in documents]
    raw: Any = encoder.predict(pairs)

    if _model_applies_activation(encoder):
        return [min(1.0, max(0.0, float(score))) for score in raw]
    return [_sigmoid(float(score)) for score in raw]


def warm_up() -> None:
    """Load the model, ignoring failure. Safe to call from startup.

    Loading is what makes the FIRST rerank slow — weights are fetched and
    deserialized inside the call. Left to happen lazily under
    `RERANK_TIMEOUT_S`, that first request times out, degrades to the fused
    order, and looks exactly like a broken reranker. Warming up separates
    "the model is loading" from "the model is too slow".
    """
    try:
        _get_cross_encoder()
    except RerankUnavailable:
        pass  # already logged; the pipeline degrades


async def rerank_async(query: str, documents: list[str]) -> list[float]:
    """Async wrapper. The forward pass is CPU-bound and blocking (CLAUDE.md).

    The timeout covers INFERENCE ONLY. Model loading happens first, outside the
    deadline, because a cold load legitimately takes minutes on a machine
    without cached weights and charging that to a per-request inference budget
    means the reranker can never successfully load under load.
    """
    if not documents:
        return []

    await asyncio.to_thread(warm_up)

    return await asyncio.wait_for(
        asyncio.to_thread(rerank_sync, query, documents),
        timeout=settings.RERANK_TIMEOUT_S,
    )
