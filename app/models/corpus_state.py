"""
Per-tenant corpus epoch — the invalidation signal for everything derived from
the corpus.

One row per user, one integer. Every corpus mutation (ingest, replace, delete)
bumps it inside the same transaction as the mutation, via
`app.services.corpus.events.corpus_changed`. The semantic cache stores the
epoch it was populated at; a lookup whose stored epoch differs from the current
one is a miss (docs/ADVANCED_RAG_PLAN.md, invariant 10).

The epoch is the CORRECTNESS layer of invalidation. The proactive Neo4j cache
delete that accompanies it is hygiene — best-effort, allowed to fail — because
a cache entry that survives the delete still cannot be served once the epoch
has moved.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class UserCorpusState(Base):
    __tablename__ = "user_corpus_state"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Monotonically increasing. 0 is the implicit epoch of a tenant with no row
    # yet — `get_corpus_epoch` returns 0 for absent rows so that a cache entry
    # written before the tenant's first mutation is still comparable.
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
