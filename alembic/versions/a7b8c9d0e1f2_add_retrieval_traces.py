"""Add retrieval_traces — persisted stage-by-stage retrieval trace.

Phase 2 of docs/ADVANCED_RAG_PLAN.md. Retrieval quality is unmeasurable without
this table: Phase 5's recall@k and nDCG are computed from `stages` and
`final_chunk_ids`.

`stages` is JSONB rather than a child table because the stage list is written
once, read whole, and its shape changes as the pipeline gains stages — exactly
the case JSONB suits. The (user_id, created_at DESC) index serves both the
per-user timeline and the aggregate metric queries.

Revision ID: a7b8c9d0e1f2
Revises: f1a2b3c4d5e6
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "retrieval_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        # No FK to users: a trace must outlive the account it describes long
        # enough to be aggregated.
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("routed_to", sa.String(length=32), nullable=False),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False),
        sa.Column("stages", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "final_chunk_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
        ),
        sa.Column("answer_hash", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_retrieval_traces_user_created",
        "retrieval_traces",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_retrieval_traces_user_created", table_name="retrieval_traces")
    op.drop_table("retrieval_traces")
