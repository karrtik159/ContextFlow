"""
Chunker registry — content_type → chunking strategy.

The extension seam for non-markdown corpora (Phase 8). Today every registered
type resolves to the structural chunker in `app.services.chunking`, because
that is the only strategy that exists; the registry's job is to make adding an
HTML or PDF chunker a registration, not a rewrite of ingestion — and to make
an UNKNOWN content type loud instead of silently pretending it is markdown.

Contract for a chunker callable: the exact signature of
`chunking.chunk_document` — `(text, *, count_tokens, target_tokens,
overlap_tokens, max_tokens, title) -> list[ChunkDraft]` — including its
invariants (no silent truncation, token budget measured on embedding_input).

`resolve_chunker` returns the DEFAULT the caller passes in, not its own import
of `chunk_document`, for the known structural types. That is deliberate: the
caller's module binding is the established test seam (ingestion tests patch
`store.chunk_document`), and the registry must not quietly bypass it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

ChunkerFn = Callable[..., list[Any]]

# Sentinel meaning "use the structural default the caller passed in".
_STRUCTURAL_DEFAULT = None

# content_type (lowercased, parameters stripped) → chunker or the sentinel.
_REGISTRY: dict[str, ChunkerFn | None] = {
    "text/plain": _STRUCTURAL_DEFAULT,
    "text/markdown": _STRUCTURAL_DEFAULT,
    "text/x-markdown": _STRUCTURAL_DEFAULT,
}


def _normalize(content_type: str) -> str:
    """Lowercase and strip parameters: 'text/plain; charset=utf-8' → 'text/plain'."""
    return (content_type or "").split(";", 1)[0].strip().lower()


def register_chunker(content_type: str, chunker: ChunkerFn) -> None:
    """Register a chunking strategy for a content type.

    Overwriting an existing registration is allowed but logged — two modules
    silently fighting over a content type is a bug worth noticing.
    """
    key = _normalize(content_type)
    if not key:
        raise ValueError("register_chunker requires a non-empty content type")
    if _REGISTRY.get(key) is not None:
        logger.warning("Chunker for %r is being replaced", key)
    _REGISTRY[key] = chunker


def resolve_chunker(content_type: str, *, default: ChunkerFn) -> ChunkerFn:
    """The chunker for this content type, or `default` with a loud log.

    `default` is the caller's own binding of the structural chunker. Known
    text types resolve to it silently; an unknown type ALSO resolves to it —
    a plain-text cut of an unknown format is degraded retrieval, not a failed
    ingest — but warns, because the degradation must be observable
    (invariant 6's spirit: never quietly).
    """
    key = _normalize(content_type)
    if key in _REGISTRY:
        registered = _REGISTRY[key]
        return registered if registered is not None else default

    logger.warning(
        "No chunker registered for content_type=%r — falling back to the "
        "structural text chunker. Headings/fences of that format will not be "
        "respected; register a chunker if this type is a real corpus source.",
        content_type,
    )
    return default
