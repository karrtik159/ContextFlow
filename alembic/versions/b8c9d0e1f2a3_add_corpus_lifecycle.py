"""Add corpus lifecycle — documents.raw_text and user_corpus_state.

Phase 7 of docs/ADVANCED_RAG_PLAN.md. Two additions, one enabler each:

`documents.raw_text` stores the ingested source verbatim. Without it the corpus
is un-reprocessable: `chunks.char_start/char_end` index a string that was thrown
away at ingest, so re-chunking after a chunker change, re-embedding after a
dimension change, and char-span highlighting were all impossible. Nullable
because pre-Phase-7 rows have no source to backfill from — they are re-ingested,
not migrated.

`user_corpus_state` carries one integer epoch per tenant, bumped inside the same
transaction as every corpus mutation. The semantic cache stores the epoch it was
populated at and a lookup compares; a corpus change therefore invalidates the
tenant's cache even when the proactive Neo4j delete fails (invariant 10).

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("raw_text", sa.Text(), nullable=True))

    op.create_table(
        "user_corpus_state",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("user_corpus_state")
    op.drop_column("documents", "raw_text")
