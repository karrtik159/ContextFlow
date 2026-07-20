"""Add documents and chunks — the retrieval corpus.

Creates the document/chunk tables that Phase 1 of docs/ADVANCED_RAG_PLAN.md
introduces, with the three indexes retrieval depends on:

  - HNSW / vector_cosine_ops on chunks.embedding  → dense retrieval
  - GIN on chunks.content_tsv                     → sparse (BM25-ish), Phase 3
  - btree (user_id, document_id)                  → tenant-scoped lookups

`chunks.user_id` is denormalized from documents.user_id ON PURPOSE. pgvector's
HNSW scan walks hnsw.ef_search global candidates and applies the WHERE clause
afterwards, so a tenant filter living on a joined table silently collapses
recall toward zero as the table grows. Keeping user_id on the chunk row is what
makes a correct tenant-scoped vector search possible at all.

`content_tsv` is a Postgres GENERATED column rather than something the
application writes, so it can never drift from `text`.

Revision ID: f1a2b3c4d5e6
Revises: d4e9f0a1b2c3
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.core.config import settings

revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "d4e9f0a1b2c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "completed", "failed", name="document_status"),
            nullable=False,
        ),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # Makes re-ingesting identical content idempotent, per user.
        sa.UniqueConstraint("user_id", "checksum", name="uq_documents_user_checksum"),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        # NOT NULL is a tenant boundary here, not a data-quality nicety.
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heading_path", sa.Text(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.Vector(settings.EMBEDDING_DIMENSIONS),
            nullable=True,
        ),
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )

    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )
    op.create_index("ix_chunks_user_document", "chunks", ["user_id", "document_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_user_document", table_name="chunks")
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_table("chunks")

    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_table("documents")

    # Enum types are not dropped with their table; leaving it behind would make
    # a re-run of upgrade() fail with "type already exists".
    sa.Enum(name="document_status").drop(op.get_bind(), checkfirst=True)
