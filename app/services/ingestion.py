"""
Backward-compatibility shim — ingestion moved to `app.services.corpus.store`.

Phase 7 made ingestion one operation of the corpus lifecycle package rather
than a standalone module, so that every corpus mutation shares one write path
and one invalidation hook (`app.services.corpus.events.corpus_changed`).

Import from `app.services.corpus` in new code. This shim only re-exports; do
not add logic here, and do not monkeypatch these names in tests — patch the
real module, `app.services.corpus.store`.
"""

from app.services.corpus.store import (  # noqa: F401
    IngestResult,
    compute_checksum,
    ingest_document,
)

__all__ = ["IngestResult", "compute_checksum", "ingest_document"]
