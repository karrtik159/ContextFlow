"""
pgvector-powered semantic search operations.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession
from app.models.message import Message


async def search_similar_messages(
    db: AsyncSession,
    query_embedding: list[float],
    user_id: UUID,
    session_id: UUID | None = None,
    limit: int = 5,
) -> list[Message]:
    """
    Find the most semantically similar messages using cosine distance.

    Args:
        db: Async database session.
        query_embedding: Embedding vector, ``settings.EMBEDDING_DIMENSIONS`` wide.
        user_id: **Required** tenant scope. Every search is confined to this
            user's sessions.
        session_id: Optional further filter to a single chat session.
        limit: Max results to return.

    Returns:
        List of Message ORM objects ordered by similarity (closest first).

    Raises:
        ValueError: If ``user_id`` is falsy. This is deliberate — the tenant
            filter is not optional. An earlier revision defaulted ``user_id``
            to ``None`` and applied the filter conditionally, which silently
            turned an unscoped call into a cross-tenant search over every
            user's messages. Failing loudly is the point: a caller with no
            user scope has no business searching this table.
    """
    if not user_id:
        raise ValueError(
            "search_similar_messages requires a user_id — unscoped vector "
            "search would read across all tenants."
        )

    stmt = (
        select(Message)
        .join(Message.session)
        .where(Message.embedding.is_not(None))
        .where(ChatSession.user_id == user_id)
        .order_by(Message.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )

    if session_id:
        stmt = stmt.where(Message.session_id == session_id)

    result = await db.execute(stmt)
    return list(result.scalars().all())
