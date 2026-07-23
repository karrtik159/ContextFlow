import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.deps import DBSession, OptionalUser, is_valid_rag_service_request, resolve_rag_user_id
from app.services.embeddings import embed_text_async
from app.services.vector_search import search_similar_messages

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context", tags=["Context"])


class PrefetchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: UUID | str | None = None
    session_id: UUID | str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class PrefetchResponse(BaseModel):
    context: str


def _parse_uuid(value: UUID | str | None) -> UUID | None:
    if value in (None, "", "anonymous"):
        return None

    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


@router.post("/prefetch", response_model=PrefetchResponse)
async def prefetch_context(
    request: PrefetchRequest,
    db: DBSession,
    http_request: Request,
    user: OptionalUser = None,
) -> Any:
    """
    Ultra-low latency semantic search for Voice AI RAG.

    This endpoint explicitly bypasses the CrewAI orchestration layer.
    It takes the STT transcript the exact millisecond the user stops speaking,
    performs a raw pgvector HNSW search, and formats the context for direct
    LLM injection via LiveKit's `on_user_turn_completed`.
    """
    is_service_request = is_valid_rag_service_request(http_request)
    resolved_user_id = resolve_rag_user_id(
        request_user_id=str(request.user_id) if request.user_id is not None else None,
        user=user,
        is_service_request=is_service_request,
    )
    if resolved_user_id is None:
        if not is_service_request and user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authentication or internal service token required",
            )
        return {"context": "No scoped conversation context is available yet."}

    user_uuid = _parse_uuid(resolved_user_id)
    session_uuid = _parse_uuid(request.session_id)
    if user_uuid is None:
        return {"context": "No scoped conversation context is available yet."}

    try:
        query_embedding = await embed_text_async(request.query)

        similar_messages = await search_similar_messages(
            db=db,
            query_embedding=query_embedding,
            session_id=session_uuid,
            user_id=user_uuid,
            limit=request.limit,
        )

        if not similar_messages:
            return {"context": "No highly relevant past context found."}

        context_lines = []
        for msg in similar_messages:
            context_lines.append(f"[{msg.role.upper()}]: {msg.content}")

        context_block = "\n".join(context_lines)
        return {"context": f"Relevant context from previous memory/conversations:\n{context_block}"}

    except Exception:
        # The real error goes to the log with the request id; the client gets
        # a generic message. Exception text can carry connection strings,
        # SQL fragments, and file paths (Phase 6).
        logger.exception("Context prefetch failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Context prefetch failed.",
        ) from None
