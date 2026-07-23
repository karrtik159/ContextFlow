"""Add chunk provenance — chunker_version and was_hard_split.

Phase 8 of docs/ADVANCED_RAG_PLAN.md. Both columns record facts ingestion
already knew and previously threw away:

`chunker_version` (invariant 11) stamps which algorithm cut the row. Server
default 1 is truthful for existing rows — everything before this migration was
cut by the version-1 algorithm. A `CHUNKER_VERSION` bump plus
scripts/rechunk_corpus.py is now a detectable, targeted re-process instead of
a full-corpus guess.

`was_hard_split` persists the mid-sentence-cut flag that was warn-logged at
ingest and then lost. A corpus-quality audit becomes a SELECT.

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c9d0e1f2a3b4"
down_revision: str | Sequence[str] | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("chunker_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "chunks",
        sa.Column("was_hard_split", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("chunks", "was_hard_split")
    op.drop_column("chunks", "chunker_version")
