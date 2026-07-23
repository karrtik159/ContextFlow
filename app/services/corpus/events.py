"""
Corpus change events — the one place invalidation happens.

`corpus_changed` is called by the API layer after every successful corpus
mutation, inside the request's transaction. It does two things, in two
different consistency classes:

1. **Bump the tenant's corpus epoch** — transactional with the mutation. This
   is the CORRECTNESS layer: the semantic cache stores the epoch it was
   populated at, lookups compare, and a mismatch is a miss. If the mutation
   rolls back, so does the bump.

2. **Delete the tenant's Neo4j cache entries** — best-effort hygiene. Neo4j is
   a different system with no shared transaction, so this can fail (or race)
   without harm: any entry the delete misses is already unservable because of
   the epoch. Failure is logged, never raised.

Future corpus-derived state (a reindex queue, a BM25 statistics refresh) hangs
its trigger here rather than in each endpoint.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.corpus_state import UserCorpusState

logger = logging.getLogger(__name__)


async def get_corpus_epoch(db: AsyncSession, *, user_id: uuid.UUID) -> int:
    """The tenant's current corpus epoch; 0 for a tenant with no mutations yet."""
    value = await db.scalar(
        select(UserCorpusState.epoch).where(UserCorpusState.user_id == user_id)
    )
    return int(value) if value is not None else 0


async def get_corpus_epoch_safe(db: AsyncSession, user_id: str) -> int:
    """Epoch lookup for the request path: never raises, never poisons.

    Two failure shapes are absorbed here. A non-UUID scope (internal service
    tokens may name arbitrary ids) has no corpus and reads as epoch 0. A
    statement error — the table missing on an unmigrated database is the live
    case — would otherwise abort the caller's whole PostgreSQL transaction
    (the `InFailedSQLTransactionError` cascade documented in
    retrieval/pipeline.py), so the query runs inside a SAVEPOINT and a failure
    degrades to epoch 0 with a warning instead of failing the request.

    Degrading to 0 is safe for correctness: populate stamps the same epoch the
    lookup compares, so the cache behaves as it did pre-Phase-7 rather than
    serving cross-epoch answers.
    """
    try:
        scoped = uuid.UUID(str(user_id))
    except (TypeError, ValueError):
        return 0

    try:
        async with db.begin_nested():
            return await get_corpus_epoch(db, user_id=scoped)
    except Exception as exc:
        logger.warning("Corpus epoch lookup failed for user=%s: %s", user_id, exc)
        return 0


async def bump_corpus_epoch(db: AsyncSession, *, user_id: uuid.UUID) -> int:
    """Atomically increment and return the tenant's epoch (first bump → 1)."""
    now = datetime.now(UTC)
    stmt = (
        pg_insert(UserCorpusState)
        .values(user_id=user_id, epoch=1, updated_at=now)
        .on_conflict_do_update(
            index_elements=[UserCorpusState.user_id],
            set_={"epoch": UserCorpusState.epoch + 1, "updated_at": now},
        )
        .returning(UserCorpusState.epoch)
    )
    return int((await db.execute(stmt)).scalar_one())


async def corpus_changed(db: AsyncSession, *, user_id: uuid.UUID) -> int:
    """The corpus mutated: bump the epoch, invalidate what derives from it.

    Returns the new epoch. Invariant 10: every write path calls this; a cache
    readable after the corpus changed under it is a bug, not a staleness
    tradeoff.
    """
    epoch = await bump_corpus_epoch(db, user_id=user_id)

    # Hygiene layer — lazy import so this module (and the unit tests over the
    # epoch logic) never require the Neo4j driver.
    try:
        from app.services.semantic_cache import invalidate_user_cache

        await invalidate_user_cache(str(user_id))
    except Exception as exc:
        logger.warning(
            "Semantic cache invalidation failed for user=%s (epoch %d still "
            "protects correctness): %s",
            user_id, epoch, exc,
        )

    logger.info("Corpus changed — user=%s epoch=%d", user_id, epoch)
    return epoch
