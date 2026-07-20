"""
RetrievalTraceRecord ORM model — the persisted stage-by-stage trace.

The trace is the product, not a debug artifact. Without it, retrieval quality is
anecdotal: there is no way to compute recall@k or nDCG after the fact, no way to
tell a cache hit from a lucky guess, and no way to see that an arm has been
returning nothing for a week. Phase 5's metrics read this table.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class RetrievalTraceRecord(Base):
    __tablename__ = "retrieval_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # No FK to users: a trace is an observability record and must survive the
    # deletion of the account it describes long enough to be aggregated. It is
    # still scoped, and every read filters on it.
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)

    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)

    routed_to: Mapped[str] = mapped_column(String(32), nullable=False)
    total_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # [{name, latency_ms, in, out, meta}] — one entry per pipeline stage.
    stages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    final_chunk_ids: Mapped[list] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    # Hash rather than the answer itself: enough to tell whether two runs
    # produced the same output, without a second copy of every response.
    answer_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_retrieval_traces_user_created", "user_id", "created_at"),
    )
