"""
Canonical embedding service — single source of truth for all vector generation.

Supports two providers (set via EMBEDDING_PROVIDER):
  - "openai"      → AsyncOpenAI / OpenAI embeddings API
  - "huggingface" → Local SentenceTransformer model

All embedding callers (semantic cache, pgvector search, CrewAI tools, context
prefetch) MUST use this module instead of ad-hoc client construction.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Thread-safe SentenceTransformer Singleton ────────────────

_hf_encoder: SentenceTransformer | None = None
_hf_encoder_lock = threading.Lock()


def _get_hf_encoder() -> SentenceTransformer:
    """Thread-safe lazy loader for the SentenceTransformer model.

    Uses a lock to prevent race conditions if multiple asyncio threads
    call this before the model is loaded. SentenceTransformer's internal
    tokenizer is NOT thread-safe for concurrent initialization.
    """
    global _hf_encoder
    if _hf_encoder is not None:
        return _hf_encoder

    with _hf_encoder_lock:
        # Double-checked locking: another thread may have loaded it while we waited
        if _hf_encoder is not None:
            return _hf_encoder

        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = settings.EMBEDDING_MODEL

        logger.info("Loading SentenceTransformer model '%s' on device '%s'...", model_name, device)
        _hf_encoder = SentenceTransformer(model_name, device=device)
        logger.info(
            "SentenceTransformer ready — dim=%d, device=%s",
            _hf_encoder.get_sentence_embedding_dimension(),
            device,
        )
    return _hf_encoder


def init_local_embedding_model() -> None:
    """Pre-load the embedding model into memory during FastAPI lifespan startup.

    Called synchronously from the lifespan factory. Validates that the loaded
    model dimension matches EMBEDDING_DIMENSIONS in config to catch .env
    mismatches early (before the first request hits Neo4j).
    """
    if settings.EMBEDDING_PROVIDER != "huggingface":
        return

    try:
        encoder = _get_hf_encoder()
        actual_dim = encoder.get_sentence_embedding_dimension()
        expected_dim = settings.EMBEDDING_DIMENSIONS

        if actual_dim != expected_dim:
            logger.error(
                "EMBEDDING_DIMENSIONS mismatch! Model '%s' produces %d-dim vectors, "
                "but config says %d. Update EMBEDDING_DIMENSIONS in .env to %d.",
                settings.EMBEDDING_MODEL, actual_dim, expected_dim, actual_dim,
            )
        else:
            logger.info("Embedding model validated — %d dimensions match config.", actual_dim)
    except Exception as exc:
        logger.error("Failed to load local embedding model: %s", exc, exc_info=True)


# ── Token Counting ──────────────────────────────────────────
#
# Lives here, not in chunking.py, because the token budget is a property of the
# configured embedding model — the same coupling that makes EMBEDDING_DIMENSIONS
# load-bearing. chunking.py takes a counter as an argument so it stays pure.

_tiktoken_encoding = None
_tiktoken_lock = threading.Lock()

# [CLS] and [SEP] are added by the HF tokenizer and consume real budget, but
# `add_special_tokens=False` is needed to count the content itself.
_HF_SPECIAL_TOKEN_ALLOWANCE = 2


def _get_tiktoken_encoding():
    """Thread-safe lazy tiktoken encoding for the configured model."""
    global _tiktoken_encoding
    if _tiktoken_encoding is not None:
        return _tiktoken_encoding

    with _tiktoken_lock:
        if _tiktoken_encoding is not None:
            return _tiktoken_encoding

        import tiktoken

        try:
            _tiktoken_encoding = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
        except KeyError:
            # Non-OpenAI models (google, openrouter, local gateways) are not in
            # tiktoken's registry. cl100k_base is an approximation — it is the
            # right encoding for the OpenAI embedding models and a reasonable
            # proxy elsewhere. Counts may be off for a genuinely different
            # tokenizer, which is why chunk budgets sit well under the ceiling.
            logger.warning(
                "No tiktoken encoding registered for '%s'; falling back to cl100k_base. "
                "Token counts are approximate for this model.",
                settings.EMBEDDING_MODEL,
            )
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoding


def count_tokens(text: str) -> int:
    """Token count under the CONFIGURED embedding model's tokenizer.

    Pass this into `chunk_document` — using a different tokenizer for chunking
    than for embedding reintroduces the silent-truncation bug that the token
    budget exists to prevent.
    """
    if not text:
        return 0

    if settings.EMBEDDING_PROVIDER == "huggingface":
        tokenizer = _get_hf_encoder().tokenizer
        # truncation=False is load-bearing. Without it the tokenizer returns at
        # most model_max_length ids, so this would measure the cap (256) instead
        # of the text and every over-budget check would silently pass.
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        return len(ids) + _HF_SPECIAL_TOKEN_ALLOWANCE

    return len(_get_tiktoken_encoding().encode(text))


def head_within_token_budget(text: str, *, max_tokens: int) -> tuple[str, bool]:
    """Largest prefix of `text` that fits `max_tokens`, and whether it was cut.

    For inputs that are NOT chunked — a whole chat message — this is the
    guard that replaces the encoder's silent truncation with a measured,
    flagged one (invariant 6). The document corpus goes through
    `chunk_document`; this is for the single-vector-per-item paths.

    Tokenizer-agnostic on purpose: it estimates a character budget from the
    measured token density, then shrinks until the prefix verifiably fits,
    rather than trusting a chars-per-token ratio that varies by script and by
    model. Same shape as chunking._hard_split, kept here because the token
    budget is a property of the embedding model this module owns.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if not text:
        return text, False

    total = count_tokens(text)
    if total <= max_tokens:
        return text, False

    chars = max(1, int(len(text) * max_tokens / max(1, total)))
    while chars > 1 and count_tokens(text[:chars]) > max_tokens:
        chars = max(1, int(chars * 0.8))
    return text[:chars], True


# ── Embedding Client Factory ────────────────────────────────


def _get_openai_embedding_client():
    """Build a sync OpenAI client for the configured embedding provider."""
    from openai import OpenAI

    from app.services.llm_provider import get_openai_compatible_kwargs

    provider = settings.EMBEDDING_PROVIDER

    if provider == "google":
        return OpenAI(
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    # "openai" and any other OpenAI-compatible provider
    return OpenAI(**get_openai_compatible_kwargs())


def _get_async_openai_embedding_client():
    """Build an async OpenAI client for the configured embedding provider."""
    from openai import AsyncOpenAI

    from app.services.llm_provider import get_openai_compatible_kwargs

    provider = settings.EMBEDDING_PROVIDER

    if provider == "google":
        return AsyncOpenAI(
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    # "openai" and any other OpenAI-compatible provider
    return AsyncOpenAI(**get_openai_compatible_kwargs())


# ── Dimension Validation ────────────────────────────────────


def _validate_embedding_dimensions(vector: list[float]) -> list[float]:
    if len(vector) != settings.EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"model returned {len(vector)} dims but the application is configured for "
            f"{settings.EMBEDDING_DIMENSIONS}. Re-embed data and migrate pgvector if you switch dimensions."
        )
    return vector


# ── Embedding API kwargs ────────────────────────────────────


def _build_embedding_kwargs(text: str | list[str]) -> dict:
    """Build the kwargs dict for an OpenAI embeddings.create() call.

    `input` accepts a single string or a batch of them.
    """
    kwargs = {
        "model": settings.EMBEDDING_MODEL,
        "input": text,
    }
    # Only explicitly pass dimensions for official OpenAI text-embedding-3 series
    # to prevent crashing local instances like LM Studio.
    if settings.EMBEDDING_PROVIDER == "openai" and "text-embedding-3" in settings.EMBEDDING_MODEL:
        kwargs["dimensions"] = settings.EMBEDDING_DIMENSIONS
    return kwargs


# ── Public API ──────────────────────────────────────────────


def embed_text(text: str) -> list[float]:
    """Synchronous embedding — used by CrewAI tools (which run in threads).

    Returns a validated, L2-normalized float vector.
    """
    if settings.EMBEDDING_PROVIDER == "huggingface":
        encoder = _get_hf_encoder()
        vec = encoder.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _validate_embedding_dimensions(vec.tolist())

    client = _get_openai_embedding_client()
    response = client.embeddings.create(**_build_embedding_kwargs(text))
    return _validate_embedding_dimensions(response.data[0].embedding)


async def embed_text_async(text: str) -> list[float]:
    """Async embedding — used by FastAPI endpoints.

    Returns a validated, L2-normalized float vector.
    """
    if settings.EMBEDDING_PROVIDER == "huggingface":
        # Offload to a thread to keep the FastAPI event loop free.
        # NOTE: SentenceTransformer.encode() holds the GIL during tokenization
        # but releases it during the PyTorch forward pass, so this does help
        # with concurrent request handling.
        return await asyncio.to_thread(embed_text, text)

    client = _get_async_openai_embedding_client()
    response = await client.embeddings.create(**_build_embedding_kwargs(text))
    return _validate_embedding_dimensions(response.data[0].embedding)


# ── Batch Embedding ─────────────────────────────────────────
#
# Document ingestion produces tens to thousands of chunks at once. Calling
# embed_text() per chunk means one HTTP round-trip each, which dominates ingest
# latency and needlessly multiplies rate-limit pressure.

# OpenAI accepts up to 2048 inputs per embeddings request. 128 is deliberately
# well under that: a batch also has an aggregate token ceiling, and smaller
# batches fail smaller when one is rejected.
_EMBED_BATCH_SIZE = 128


def _order_batch_response(response, expected: int) -> list[list[float]]:
    """Return embeddings in input order.

    The API returns an `index` on each item; relying on positional order would
    silently mis-pair vectors with chunks if that ever changed. A mis-paired
    corpus is close to undetectable downstream, so this sorts explicitly.
    """
    items = sorted(response.data, key=lambda d: d.index)
    if len(items) != expected:
        raise ValueError(
            f"Embedding batch returned {len(items)} vectors for {expected} inputs."
        )
    return [_validate_embedding_dimensions(item.embedding) for item in items]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Synchronous batch embedding — for CrewAI tools and worker threads.

    Returns vectors in the same order as `texts`.
    """
    if not texts:
        return []

    if settings.EMBEDDING_PROVIDER == "huggingface":
        encoder = _get_hf_encoder()
        vectors = encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        return [_validate_embedding_dimensions(v.tolist()) for v in vectors]

    client = _get_openai_embedding_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(**_build_embedding_kwargs(batch))
        out.extend(_order_batch_response(response, len(batch)))
    return out


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    """Async batch embedding — for FastAPI endpoints.

    Returns vectors in the same order as `texts`.
    """
    if not texts:
        return []

    if settings.EMBEDDING_PROVIDER == "huggingface":
        return await asyncio.to_thread(embed_texts, texts)

    client = _get_async_openai_embedding_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        response = await client.embeddings.create(**_build_embedding_kwargs(batch))
        out.extend(_order_batch_response(response, len(batch)))
    return out


async def embed_text_async_safe(text: str) -> list[float] | None:
    """Like embed_text_async but returns None on failure instead of raising.

    Use this for best-effort flows like semantic caching where a missing
    embedding should gracefully degrade rather than crash the request.
    """
    try:
        return await embed_text_async(text)
    except Exception as exc:
        logger.warning("Embedding generation failed (%s), skipping", exc)
        return None
