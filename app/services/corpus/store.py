"""
Corpus store — crack, chunk, embed, index; list, replace, delete.

The write half of the RAG pipeline, moved here from `app/services/ingestion.py`
(which remains as an import shim). Given raw text and an owning user, ingestion
produces a `Document` and its `Chunk` rows, each embedded and ready for
retrieval. The lifecycle functions complete the loop the corpus previously
lacked: before Phase 7 it was append-only, so an edited document became a
*second* document and both versions were retrieved side by side.

Tenant scope is a required argument throughout, never inferred and never
optional (docs/ADVANCED_RAG_PLAN.md §5.2). `Chunk.user_id` is written on every
row: it is the column the retrieval filter runs against, and a chunk without it
would be invisible to its owner and visible to a scan.

None of these functions call `corpus_changed` themselves — the epoch bump
belongs to the caller that owns the transaction (the API layer), so that a
rolled-back mutation does not leave a bumped epoch behind.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Chunk, Document
from app.services.chunking import chunk_document
from app.services.embeddings import count_tokens, embed_texts_async

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    chunk_count: int
    token_count: int
    was_deduplicated: bool
    hard_split_chunks: int


def compute_checksum(text: str) -> str:
    """SHA-256 of the document body, used for per-user idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def ingest_document(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    text: str,
    title: str | None = None,
    source_uri: str | None = None,
    content_type: str = "text/plain",
) -> IngestResult:
    """Ingest one document for one user.

    Idempotent on (user_id, checksum): re-ingesting identical content returns
    the existing document rather than duplicating the corpus.

    Raises:
        ValueError: if `text` is blank, or if chunking produces a chunk over
            EMBEDDING_MAX_TOKENS. The latter must fail the ingest rather than
            store a row whose embedding silently represents only its prefix.
    """
    if not text.strip():
        raise ValueError("Cannot ingest an empty document.")

    checksum = compute_checksum(text)

    existing = await db.scalar(
        select(Document).where(Document.user_id == user_id, Document.checksum == checksum)
    )
    if existing is not None:
        chunk_count = len(await _existing_chunk_ids(db, existing.id, user_id))
        logger.info(
            "Document already ingested — user=%s checksum=%s document=%s",
            user_id, checksum[:12], existing.id,
        )
        return IngestResult(
            document_id=existing.id,
            chunk_count=chunk_count,
            token_count=existing.token_count or 0,
            was_deduplicated=True,
            hard_split_chunks=0,
        )

    drafts = chunk_document(
        text,
        count_tokens=count_tokens,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        max_tokens=settings.EMBEDDING_MAX_TOKENS,
        title=title,
    )
    if not drafts:
        raise ValueError("Document produced no chunks — it may contain no extractable text.")

    # Belt and braces: chunk_document already raises past the ceiling, but this
    # is the last point before the encoder sees the text, and MiniLM truncates
    # without complaint. An assertion here is cheaper than a corpus of rows
    # whose vectors represent only their first 200 words.
    for draft in drafts:
        if draft.token_count > settings.EMBEDDING_MAX_TOKENS:
            raise ValueError(
                f"Chunk at [{draft.char_start}:{draft.char_end}] is {draft.token_count} "
                f"tokens, over the {settings.EMBEDDING_MAX_TOKENS}-token ceiling for "
                f"{settings.EMBEDDING_MODEL}."
            )

    hard_split = sum(1 for d in drafts if d.was_hard_split)
    if hard_split:
        # Loud rather than silent: a hard split means we cut mid-sentence
        # because a unit had no internal structure, which degrades retrieval.
        logger.warning(
            "Ingest hard-split %d/%d chunks mid-sentence — user=%s title=%r. "
            "The source likely contains very long unbroken text.",
            hard_split, len(drafts), user_id, title,
        )

    document = Document(
        user_id=user_id,
        source_uri=source_uri,
        title=title,
        content_type=content_type,
        checksum=checksum,
        status="processing",
        token_count=count_tokens(text),
        # The verbatim source. `char_start/char_end` index this string, and
        # re-chunking or re-embedding the corpus later is only possible
        # because it was kept.
        raw_text=text,
    )
    db.add(document)
    await db.flush()  # assign document.id without committing

    # One batched call, not one call per chunk.
    vectors = await embed_texts_async([d.embedding_input for d in drafts])
    if len(vectors) != len(drafts):
        raise ValueError(
            f"Embedding returned {len(vectors)} vectors for {len(drafts)} chunks."
        )

    db.add_all(
        [
            Chunk(
                document_id=document.id,
                # Denormalized deliberately — see the comment on Chunk.user_id.
                user_id=user_id,
                chunk_index=index,
                text=draft.text,
                heading_path=draft.heading_path,
                char_start=draft.char_start,
                char_end=draft.char_end,
                token_count=draft.token_count,
                embedding=vector,
            )
            for index, (draft, vector) in enumerate(zip(drafts, vectors))
        ]
    )

    document.status = "completed"
    await db.flush()

    logger.info(
        "Ingested document=%s user=%s chunks=%d tokens=%d hard_split=%d",
        document.id, user_id, len(drafts), document.token_count, hard_split,
    )

    return IngestResult(
        document_id=document.id,
        chunk_count=len(drafts),
        token_count=document.token_count or 0,
        was_deduplicated=False,
        hard_split_chunks=hard_split,
    )


async def list_documents(
    db: AsyncSession, *, user_id: uuid.UUID
) -> list[tuple[Document, int]]:
    """This tenant's documents, newest first, each with its chunk count.

    The chunk count comes from an outer-join aggregate rather than
    `len(document.chunks)` so listing N documents is one query, not N lazy
    loads — and so a document whose chunks failed to write shows 0 rather
    than raising.
    """
    _require_user_id(user_id, "list_documents")

    stmt = (
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .where(Document.user_id == user_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [(document, int(count)) for document, count in rows]


async def delete_document(
    db: AsyncSession, *, user_id: uuid.UUID, document_id: uuid.UUID
) -> bool:
    """Delete one document and (via FK cascade) all of its chunks.

    Returns False when no document with this id exists FOR THIS TENANT —
    another tenant's id and a nonexistent id are deliberately
    indistinguishable, so the API layer can 404 both without confirming
    existence across the boundary.
    """
    _require_user_id(user_id, "delete_document")

    document = await db.scalar(
        select(Document).where(Document.id == document_id, Document.user_id == user_id)
    )
    if document is None:
        return False

    await db.delete(document)
    await db.flush()
    logger.info("Deleted document=%s user=%s", document_id, user_id)
    return True


async def replace_document(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    text: str,
    title: str | None = None,
    source_uri: str | None = None,
    content_type: str = "text/plain",
) -> IngestResult | None:
    """Replace a document's content wholly: delete + ingest, one transaction.

    Returns None when the document does not exist for this tenant (the API
    layer's 404), mirroring `delete_document`.

    Replacement is total — `title`/`source_uri`/`content_type` are taken as
    given, not inherited from the old version. There is no version chain:
    nothing in the product needs history, and a supersede graph is complexity
    the retrieval filter would then have to know about.

    If the new text is identical to ANOTHER existing document's content, the
    idempotency rule wins: the old document is still deleted and the result
    points at the existing duplicate (`was_deduplicated=True`).
    """
    _require_user_id(user_id, "replace_document")

    document = await db.scalar(
        select(Document).where(Document.id == document_id, Document.user_id == user_id)
    )
    if document is None:
        return None

    # Delete first and flush, so re-ingesting identical content builds a fresh
    # document instead of dedup-matching the row that is about to disappear.
    await db.delete(document)
    await db.flush()

    result = await ingest_document(
        db,
        user_id=user_id,
        text=text,
        title=title,
        source_uri=source_uri,
        content_type=content_type,
    )
    logger.info(
        "Replaced document=%s with document=%s user=%s",
        document_id, result.document_id, user_id,
    )
    return result


def _require_user_id(user_id: uuid.UUID | None, fn: str) -> uuid.UUID:
    if not user_id:
        raise ValueError(
            f"{fn} requires a user_id — an unscoped corpus operation would touch all tenants."
        )
    return user_id


async def _existing_chunk_ids(
    db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> list[uuid.UUID]:
    """Chunk ids for a document, scoped to its owner.

    The user_id predicate is redundant given document_id is already owned, and
    that is the point: the tenant filter is unconditional on every chunk query
    so it cannot be forgotten on the one that matters.
    """
    result = await db.scalars(
        select(Chunk.id).where(Chunk.document_id == document_id, Chunk.user_id == user_id)
    )
    return list(result)
