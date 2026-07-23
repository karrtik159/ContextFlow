"""
Re-chunk and re-embed documents cut by an outdated chunker version.

Usage:
    uv run python scripts/rechunk_corpus.py            # report only (dry run)
    uv run python scripts/rechunk_corpus.py --apply    # actually re-process
    uv run python scripts/rechunk_corpus.py --apply --user <uuid>

Selects documents that have `raw_text` (Phase 7 onward — older rows have no
source to re-process and must be re-ingested) and at least one chunk whose
`chunker_version` lags `chunking.CHUNKER_VERSION`. For each: delete the
chunks, re-chunk via the registry, re-embed in one batch, insert.

Each affected tenant gets ONE `corpus_changed` at the end — re-chunking is a
corpus mutation like any other (invariant 10), and their cached answers cite
chunk ids that no longer exist.

Deliberately serial, one document per transaction: this is an offline
maintenance script where crash-resumability (already-processed documents stay
processed) matters more than wall clock.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rechunk_corpus")


async def _find_outdated(session, user_filter: uuid.UUID | None) -> list[uuid.UUID]:
    from sqlalchemy import exists, select

    from app.models.document import Chunk, Document
    from app.services.chunking import CHUNKER_VERSION

    outdated_chunk = exists(
        select(1).where(
            Chunk.document_id == Document.id,
            Chunk.chunker_version < CHUNKER_VERSION,
        )
    )
    stmt = select(Document.id).where(Document.raw_text.is_not(None), outdated_chunk)
    if user_filter is not None:
        stmt = stmt.where(Document.user_id == user_filter)
    return list(await session.scalars(stmt))


async def _rechunk_one(session, document_id: uuid.UUID) -> tuple[uuid.UUID, int]:
    """Re-process one document. Returns (owner user_id, new chunk count)."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy import select

    from app.core.config import settings
    from app.models.document import Chunk, Document
    from app.services.chunkers import resolve_chunker
    from app.services.chunking import CHUNKER_VERSION, chunk_document
    from app.services.embeddings import count_tokens, embed_texts_async

    document = await session.scalar(select(Document).where(Document.id == document_id))
    if document is None or not document.raw_text:
        raise ValueError(f"Document {document_id} vanished or has no raw_text")

    chunker = resolve_chunker(document.content_type, default=chunk_document)
    drafts = chunker(
        document.raw_text,
        count_tokens=count_tokens,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        max_tokens=settings.EMBEDDING_MAX_TOKENS,
        title=document.title,
    )
    if not drafts:
        raise ValueError(f"Document {document_id} produced no chunks on re-chunk")

    vectors = await embed_texts_async([d.embedding_input for d in drafts])
    if len(vectors) != len(drafts):
        raise ValueError(f"Embedding returned {len(vectors)} vectors for {len(drafts)} chunks")

    # Delete-then-insert inside one transaction: a reader mid-transaction sees
    # the old chunks or the new ones, never neither.
    await session.execute(
        sa_delete(Chunk).where(
            Chunk.document_id == document.id, Chunk.user_id == document.user_id
        )
    )
    session.add_all(
        [
            Chunk(
                document_id=document.id,
                user_id=document.user_id,
                chunk_index=index,
                text=draft.text,
                heading_path=draft.heading_path,
                char_start=draft.char_start,
                char_end=draft.char_end,
                token_count=draft.token_count,
                chunker_version=CHUNKER_VERSION,
                was_hard_split=draft.was_hard_split,
                embedding=vector,
            )
            for index, (draft, vector) in enumerate(zip(drafts, vectors))
        ]
    )
    return document.user_id, len(drafts)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually re-process (default: report only)")
    parser.add_argument("--user", type=uuid.UUID, default=None, help="Limit to one tenant")
    args = parser.parse_args()

    from app.core.db import async_session
    from app.services.chunking import CHUNKER_VERSION
    from app.services.corpus import corpus_changed

    async with async_session() as session:
        outdated = await _find_outdated(session, args.user)

    logger.info(
        "%d document(s) below chunker version %d%s",
        len(outdated), CHUNKER_VERSION, "" if args.apply else " (dry run — pass --apply)",
    )
    if not args.apply or not outdated:
        return 0

    affected_users: set[uuid.UUID] = set()
    failures = 0
    for document_id in outdated:
        try:
            async with async_session() as session:
                async with session.begin():
                    owner, count = await _rechunk_one(session, document_id)
            affected_users.add(owner)
            logger.info("Re-chunked %s → %d chunks", document_id, count)
        except Exception:
            failures += 1
            logger.exception("FAILED to re-chunk %s — continuing", document_id)

    for owner in sorted(affected_users, key=str):
        async with async_session() as session:
            async with session.begin():
                epoch = await corpus_changed(session, user_id=owner)
        logger.info("Bumped corpus epoch for user=%s → %d", owner, epoch)

    logger.info("Done: %d re-chunked, %d failed", len(outdated) - failures, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
