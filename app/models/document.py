"""
Document and Chunk ORM models — the retrieval corpus.

Before this, the only embedded entity in the system was `Message.content`:
whole chat messages, one vector each. That is conversational memory search, not
retrieval over a document corpus. These two tables are the corpus.

A Document is an ingested source (a file, a pasted body of text, a fetched URI).
A Chunk is a retrievable span of one document, carrying its own embedding and
its own full-text vector.
"""

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.db import Base

DocumentStatus = Enum(
    "pending",
    "processing",
    "completed",
    "failed",
    name="document_status",
)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False, default="text/plain")

    # SHA-256 of the raw body. Paired with user_id in a unique constraint so
    # re-ingesting the same document is idempotent rather than duplicating the
    # corpus. Scoped per user, not global: two tenants uploading identical
    # content each own their own copy, and neither can probe for the other's.
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(DocumentStatus, nullable=False, default="pending")
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    __table_args__ = (UniqueConstraint("user_id", "checksum", name="uq_documents_user_checksum"),)

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # DENORMALIZED FROM documents.user_id — DELIBERATELY. Do not "normalize"
    # this away by joining to documents for the tenant filter.
    #
    # pgvector's HNSW scan walks `hnsw.ef_search` global candidates and applies
    # the WHERE clause AFTER. If the tenant filter lives on a joined table, a
    # user whose chunks are not in that global candidate set gets ZERO rows —
    # recall collapses toward 0 as the table grows, silently. Keeping user_id on
    # the same row makes a partial-index / iterative-scan strategy possible.
    #
    # This is also a tenant boundary: NOT NULL, and every query MUST filter on
    # it. See docs/ADVANCED_RAG_PLAN.md §5 invariants 1-3.
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # "Title > Section > Subsection" — prepended to the embedded text so an
    # isolated chunk carries the context its position gave it.
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSIONS), nullable=True
    )

    # Generated in Postgres rather than in Python so it can never drift from
    # `text`. Feeds the BM25-ish sparse retrieval arm added in Phase 3.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_chunks_user_document", "user_id", "document_id"),
    )

    document = relationship("Document", back_populates="chunks")
