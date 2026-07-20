"""
Document ingestion endpoint — builds the retrieval corpus.

Scope rules match /rag/query exactly (app/api/deps.py::resolve_rag_user_id):
a JWT user's identity wins, an internal service token may name any user, and an
anonymous caller is rejected. Documents are user-owned, so there is no unscoped
ingest — a document with no owner could never be retrieved by anyone, and
accepting one would be a write path with no authentication.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.deps import DBSession, OptionalUser, is_valid_rag_service_request, resolve_rag_user_id
from app.services.ingestion import ingest_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Guards a single request from producing an unbounded embedding bill. Larger
# corpora belong in a batch/offline path, not a synchronous POST.
MAX_DOCUMENT_CHARS = 1_000_000


class DocumentIngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    title: str | None = Field(default=None, max_length=512)
    source_uri: str | None = None
    content_type: str = Field(default="text/plain", max_length=128)
    user_id: str | None = Field(
        default=None,
        description="Owning user. Only an internal service caller may set this.",
    )


class DocumentIngestResponse(BaseModel):
    document_id: uuid.UUID
    chunk_count: int
    token_count: int
    deduplicated: bool = Field(
        description="True when identical content was already ingested for this user."
    )
    hard_split_chunks: int = Field(
        description="Chunks cut mid-sentence because the source had no usable structure."
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentIngestResponse,
    summary="Ingest a document into the retrieval corpus",
)
async def ingest(
    request: DocumentIngestRequest,
    db: DBSession,
    http_request: Request,
    user: OptionalUser = None,
) -> DocumentIngestResponse:
    """Crack, chunk, embed, and index a document under the caller's scope."""
    resolved_user_id = resolve_rag_user_id(
        request_user_id=request.user_id,
        user=user,
        is_service_request=is_valid_rag_service_request(http_request),
    )
    if resolved_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication or an internal service token is required to ingest documents.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        owner_id = uuid.UUID(str(resolved_user_id))
    except (TypeError, ValueError):
        # Fail closed. A scope we cannot parse must not widen into an unscoped
        # write (docs/ADVANCED_RAG_PLAN.md §5.3).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resolved user scope is not a valid identifier.",
        ) from None

    try:
        result = await ingest_document(
            db,
            user_id=owner_id,
            text=request.text,
            title=request.title,
            source_uri=request.source_uri,
            content_type=request.content_type,
        )
    except ValueError as exc:
        # ValueError here is a rejected document (blank, or a chunk over the
        # model ceiling), not a server fault — and its message describes the
        # caller's input, so it is safe to return.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DocumentIngestResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        token_count=result.token_count,
        deduplicated=result.was_deduplicated,
        hard_split_chunks=result.hard_split_chunks,
    )
