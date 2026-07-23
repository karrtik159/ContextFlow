"""Messages-arm parity — denormalized user_id and embedding_truncated.

Phase 9 of docs/ADVANCED_RAG_PLAN.md. The messages retrieval arm filtered
tenant scope through a JOIN to chat_sessions, because messages carried no
user_id of their own. That is the same HNSW pre-filter weakness documented on
chunks.user_id: pgvector walks its global candidate set and applies the WHERE
afterwards, so a tenant whose messages are not in the global top-N gets zero
rows, and recall erodes silently as the table grows. Chunks were denormalized
from the start; messages were deferred out of Phase 2. This closes it.

`user_id` is added nullable, backfilled from the owning session, then made NOT
NULL — the standard expand pattern. The FK from messages.session_id to
chat_sessions guarantees every message has an owner, so the backfill covers
every row.

`embedding_truncated` records that a message longer than the embedding model's
token ceiling was embedded from a budgeted head rather than silently truncated
by the encoder (invariant 6). Server default false is truthful for existing
rows: they were embedded whole by the old path, and any silent truncation there
is not something this column can retroactively know about.

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "d0e1f2a3b4c5"
down_revision: str | Sequence[str] | None = "c9d0e1f2a3b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add nullable, so the column can exist before it is populated.
    op.add_column(
        "messages",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    # 2. Backfill from the owning session. The FK guarantees a match for every
    #    row, so no message is left with a NULL owner.
    op.execute(
        """
        UPDATE messages AS m
        SET user_id = cs.user_id
        FROM chat_sessions AS cs
        WHERE m.session_id = cs.id
        """
    )

    # 3. Enforce the tenant boundary: NOT NULL now that every row has an owner.
    op.alter_column("messages", "user_id", nullable=False)

    # 4. Index the filter column — same reason chunks has one: it makes a
    #    partial-index / iterative-scan strategy possible for the HNSW arm.
    op.create_index("ix_messages_user_id", "messages", ["user_id"])

    op.add_column(
        "messages",
        sa.Column(
            "embedding_truncated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("messages", "embedding_truncated")
    op.drop_index("ix_messages_user_id", table_name="messages")
    op.drop_column("messages", "user_id")
