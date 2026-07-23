"""
Document lifecycle endpoints — the corpus's only write path.

Scope rules match /rag/query exactly (app/api/deps.py::resolve_rag_user_id):
a JWT user's identity wins, an internal service token may name any user, and an
anonymous caller is rejected. Documents are user-owned, so there is no unscoped
operation — a document with no owner could never be retrieved by anyone.

Every mutation ends with `corpus_changed` (Phase 7, invariant 10): the epoch
bump rides the request transaction, so a rollback rolls the bump back too, and
the semantic cache can never serve an answer grounded in content that this
request removed.

Lookups by id return 404 for another tenant's document, not 403 — the store
makes "not yours" and "not there" indistinguishable on purpose, so existence
is never confirmed across the tenant boundary.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.api.deps import DBSession, OptionalUser, is_valid_rag_service_request, resolve_rag_user_id
from app.services.corpus import (
    corpus_changed,
    delete_document,
    ingest_document,
    list_documents,
    replace_document,
)

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


class DocumentSummary(BaseModel):
    document_id: uuid.UUID
    title: str | None
    source_uri: str | None
    content_type: str
    status: str
    token_count: int | None
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class DocumentReplaceRequest(BaseModel):
    """Total replacement — nothing is inherited from the old version."""

    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    title: str | None = Field(default=None, max_length=512)
    source_uri: str | None = None
    content_type: str = Field(default="text/plain", max_length=128)
    user_id: str | None = Field(
        default=None,
        description="Owning user. Only an internal service caller may set this.",
    )


def _resolve_owner_id(
    request_user_id: str | None,
    user: dict | None,
    http_request: Request,
) -> uuid.UUID:
    """Resolve and parse the tenant scope, or raise. Shared by every endpoint."""
    resolved_user_id = resolve_rag_user_id(
        request_user_id=request_user_id,
        user=user,
        is_service_request=is_valid_rag_service_request(http_request),
    )
    if resolved_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication or an internal service token is required for document operations.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return uuid.UUID(str(resolved_user_id))
    except (TypeError, ValueError):
        # Fail closed. A scope we cannot parse must not widen into an unscoped
        # operation (docs/ADVANCED_RAG_PLAN.md §5.3).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resolved user scope is not a valid identifier.",
        ) from None


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
    owner_id = _resolve_owner_id(request.user_id, user, http_request)

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

    # A dedup hit wrote nothing, so nothing derived from the corpus changed —
    # bumping the epoch would evict the tenant's cache for a no-op request.
    if not result.was_deduplicated:
        await corpus_changed(db, user_id=owner_id)

    return DocumentIngestResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        token_count=result.token_count,
        deduplicated=result.was_deduplicated,
        hard_split_chunks=result.hard_split_chunks,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List the caller's documents",
)
async def list_owned(
    db: DBSession,
    http_request: Request,
    user: OptionalUser = None,
    user_id: str | None = None,
) -> DocumentListResponse:
    owner_id = _resolve_owner_id(user_id, user, http_request)

    rows = await list_documents(db, user_id=owner_id)
    return DocumentListResponse(
        documents=[
            DocumentSummary(
                document_id=document.id,
                title=document.title,
                source_uri=document.source_uri,
                content_type=document.content_type,
                status=document.status,
                token_count=document.token_count,
                chunk_count=chunk_count,
                created_at=document.created_at,
            )
            for document, chunk_count in rows
        ]
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete(
    document_id: uuid.UUID,
    db: DBSession,
    http_request: Request,
    user: OptionalUser = None,
    user_id: str | None = None,
) -> Response:
    owner_id = _resolve_owner_id(user_id, user, http_request)

    deleted = await delete_document(db, user_id=owner_id, document_id=document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    await corpus_changed(db, user_id=owner_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put(
    "/{document_id}",
    response_model=DocumentIngestResponse,
    summary="Replace a document's content wholly",
)
async def replace(
    document_id: uuid.UUID,
    request: DocumentReplaceRequest,
    db: DBSession,
    http_request: Request,
    user: OptionalUser = None,
) -> DocumentIngestResponse:
    owner_id = _resolve_owner_id(request.user_id, user, http_request)

    try:
        result = await replace_document(
            db,
            user_id=owner_id,
            document_id=document_id,
            text=request.text,
            title=request.title,
            source_uri=request.source_uri,
            content_type=request.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # Unlike ingest, a replace ALWAYS mutated the corpus when it found the
    # document — the old version was deleted even when the new text
    # dedup-matched other existing content.
    await corpus_changed(db, user_id=owner_id)

    return DocumentIngestResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        token_count=result.token_count,
        deduplicated=result.was_deduplicated,
        hard_split_chunks=result.hard_split_chunks,
    )
