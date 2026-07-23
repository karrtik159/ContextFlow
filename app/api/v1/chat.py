"""
Chat endpoints — sessions & messages.

Uses *Internal schemas for FastCRUD create() which requires .model_dump().
"""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DBSession
from app.core.config import settings
from app.models.chat_session import ChatSession
from app.schemas.chat_session import SessionCreate, SessionCreateInternal, SessionRead
from app.schemas.message import MessageCreate, MessageCreateInternal, MessageRead
from app.services.crud import message_crud, session_crud
from app.services.embeddings import embed_text_async_safe, head_within_token_budget

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/sessions", response_model=SessionRead, status_code=201)
async def create_session(user_id: UUID, payload: SessionCreate, db: DBSession):
    """Start a new chat session for a user."""
    internal = SessionCreateInternal(
        title=payload.title,
        user_id=user_id,
    )
    new_session = await session_crud.create(
        db,
        object=internal,
        schema_to_select=SessionRead,
        return_as_model=True,
    )
    return new_session


@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def list_messages(session_id: UUID, db: DBSession):
    """Retrieve all messages in a session."""
    result = await message_crud.get_multi(
        db,
        schema_to_select=MessageRead,
        return_as_model=True,
        session_id=session_id,
    )
    # get_multi returns {"data": [...], "total_count": N}
    return result["data"]


@router.post("/sessions/{session_id}/messages", response_model=MessageRead, status_code=201)
async def create_message(session_id: UUID, payload: MessageCreate, db: DBSession):
    """Add a new message to a session.

    The message's tenant scope is the SESSION's owner, resolved server-side and
    denormalized onto the row (Phase 9) — never client-supplied. The messages
    retrieval arm filters on `messages.user_id` directly, so a row without it
    would be invisible to its owner and visible to a scan.
    """
    session = await db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found."
        )

    embedding = None
    truncated = False
    if payload.role in {"user", "assistant"}:
        # Bound the embedding input to the model ceiling here rather than let
        # the encoder truncate silently (invariant 6). A message over budget
        # embeds its head, and the row records that it did.
        head, truncated = head_within_token_budget(
            payload.content, max_tokens=settings.EMBEDDING_MAX_TOKENS
        )
        if truncated:
            logger.warning(
                "Message in session=%s exceeded the %d-token embedding ceiling; "
                "embedded a budgeted head and flagged embedding_truncated.",
                session_id, settings.EMBEDDING_MAX_TOKENS,
            )
        embedding = await embed_text_async_safe(head)

    internal = MessageCreateInternal(
        role=payload.role,
        content=payload.content,
        session_id=session_id,
        user_id=session.user_id,
        embedding=embedding,
        embedding_truncated=truncated,
    )
    new_msg = await message_crud.create(
        db,
        object=internal,
        schema_to_select=MessageRead,
        return_as_model=True,
    )
    return new_msg
