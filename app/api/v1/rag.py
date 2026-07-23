"""
RAG endpoint — deterministic retrieval with semantic intent routing.

The orchestration lives in app/services/rag_service.py and
app/services/retrieval/. This module is transport: resolve scope, route, persist
the trace, return. `routed_to` remains the observability contract —
cache | direct | crewai | direct_fallback.
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import DBSession, OptionalUser, is_valid_rag_service_request, resolve_rag_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="The user's question.")
    user_id: str | None = Field(default=None, description="Optional user ID for personalized context.")
    stream: bool = Field(default=False, description="If true, stream the response for simple chat.")


class Citation(BaseModel):
    label: str
    source: str
    chunk_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    heading_path: str | None = None
    score: float


class RAGQueryResponse(BaseModel):
    answer: str
    query: str
    user_id: str | None = None
    routed_to: str = Field(description="'cache' for semantic hit, 'direct' for simple chat, 'crewai' for RAG queries.")
    trace_id: uuid.UUID | None = None
    citations: list[Citation] = Field(default_factory=list)


def _process_memory_background(query: str, answer: str, user_id: str):
    """Fire-and-forget: extract entities from the Q&A and persist to graph."""
    import time

    from agents.crews.memory_crew import MemoryCrew

    transcript = f"User: {query}\nAssistant: {answer}"
    query_snippet = query[:80]

    logger.info("MemoryCrew started — user=%s query='%s'", user_id, query_snippet)
    t0 = time.perf_counter()

    try:
        result = MemoryCrew(user_id=user_id).crew().kickoff(inputs={"transcript": transcript})
        logger.info(
            "MemoryCrew completed in %.1fs — user=%s query='%s' result_len=%d",
            time.perf_counter() - t0, user_id, query_snippet, len(str(result)),
        )
    except Exception:
        logger.exception(
            "MemoryCrew FAILED after %.1fs — user=%s query='%s'",
            time.perf_counter() - t0, user_id, query_snippet,
        )


@router.post(
    "/query",
    status_code=status.HTTP_200_OK,
    response_model=RAGQueryResponse,
    summary="Ask a question — auto-routed between direct chat and full RAG",
)
async def rag_query(
    request: RAGQueryRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    db: DBSession,
    user: OptionalUser = None,
):
    """Smart-routed query endpoint.

    0. Cache-key normalization (aggressive) and retrieval normalization (light).
    1. Intent classification — cheap, fails open to RAG.
    2. User-scoped semantic cache; a hit short-circuits everything.
    3. Simple chat bypasses retrieval entirely.
    4. Knowledge queries run the deterministic pipeline, then ONE synthesis call.
    """
    from app.services.corpus import get_corpus_epoch_safe
    from app.services.embeddings import embed_text_async_safe
    from app.services.llm_provider import classify_intent, direct_chat, stream_direct_chat
    from app.services.query_normalizer import normalize_for_cache_key, normalize_for_retrieval
    from app.services.rag_service import answer_knowledge_query
    from app.services.semantic_cache import get_cached_response, populate_semantic_cache
    from app.services.trace_store import persist_trace

    resolved_user_id = resolve_rag_user_id(
        request_user_id=request.user_id,
        user=user,
        is_service_request=is_valid_rag_service_request(http_request),
    )

    # Every store this endpoint reads is user-owned, so an unscoped caller could
    # only return nothing or read across tenants — it previously did the latter.
    if resolved_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication or an internal service token is required for RAG queries.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Two different normalizations for two different jobs — see query_normalizer.
    cache_key_query = normalize_for_cache_key(request.query)
    retrieval_query = normalize_for_retrieval(request.query)

    needs_rag = await classify_intent(request.query)

    query_embedding = None
    if needs_rag:
        # Resolved ONCE per request: the cache lookup compares against it and
        # the populate task stamps it, so a corpus mutation mid-request cannot
        # label a stale answer current (Phase 7, invariant 10).
        corpus_epoch = await get_corpus_epoch_safe(db, resolved_user_id)
        # Embedded ONCE, reused by the cache lookup and every dense arm. The
        # old ReAct loop re-embedded per tool call.
        query_embedding = await embed_text_async_safe(retrieval_query)
        if query_embedding:
            cached_answer = await get_cached_response(
                normalized_query=cache_key_query,
                embedding=query_embedding,
                user_id=resolved_user_id,
                corpus_epoch=corpus_epoch,
            )
            if cached_answer:
                logger.info("Semantic Cache Hit: user=%s query='%s'", resolved_user_id, request.query[:80])
                return RAGQueryResponse(
                    answer=cached_answer,
                    query=request.query,
                    user_id=resolved_user_id,
                    routed_to="cache",
                )

    # ── Simple chat ─────────────────────────────────────────
    if not needs_rag:
        logger.info("Intent: simple chat — bypassing retrieval for '%s'", request.query[:80])

        if request.stream:
            collected: list[str] = []

            async def _stream_generator():
                async for chunk in stream_direct_chat(request.query):
                    collected.append(chunk)
                    yield chunk
                background_tasks.add_task(
                    _process_memory_background,
                    query=request.query,
                    answer="".join(collected),
                    user_id=resolved_user_id,
                )

            return StreamingResponse(_stream_generator(), media_type="text/plain")

        answer = await direct_chat(request.query)
        background_tasks.add_task(
            _process_memory_background,
            query=request.query, answer=answer, user_id=resolved_user_id,
        )
        return RAGQueryResponse(
            answer=answer, query=request.query, user_id=resolved_user_id, routed_to="direct",
        )

    # ── Knowledge query ─────────────────────────────────────
    if not query_embedding:
        # No embedding means no dense retrieval. Degrade honestly rather than
        # running a pipeline whose main arm cannot participate.
        logger.warning("Embedding unavailable; degrading to direct chat for '%s'", request.query[:80])
        answer = await direct_chat(request.query)
        return RAGQueryResponse(
            answer=answer, query=request.query, user_id=resolved_user_id,
            routed_to="direct_fallback",
        )

    outcome = await answer_knowledge_query(
        db,
        user_id=resolved_user_id,
        original_query=request.query,
        retrieval_query=retrieval_query,
        query_embedding=query_embedding,
    )

    background_tasks.add_task(
        _process_memory_background,
        query=request.query, answer=outcome.answer, user_id=resolved_user_id,
    )
    if outcome.cacheable:
        background_tasks.add_task(
            populate_semantic_cache,
            normalized_query=cache_key_query,
            embedding=query_embedding,
            answer=outcome.answer,
            user_id=resolved_user_id,
            session_id=None,
            corpus_epoch=corpus_epoch,
        )
    if outcome.trace is not None:
        background_tasks.add_task(
            persist_trace,
            trace=outcome.trace,
            routed_to=outcome.routed_to,
            answer_hash=outcome.answer_hash,
        )

    citations = [
        Citation(
            label=chunk.citation_label,
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            heading_path=chunk.heading_path,
            score=chunk.score,
        )
        for chunk in (outcome.trace.final_chunks if outcome.trace else [])
    ]

    return RAGQueryResponse(
        answer=outcome.answer,
        query=request.query,
        user_id=resolved_user_id,
        routed_to=outcome.routed_to,
        trace_id=outcome.trace.trace_id if outcome.trace else None,
        citations=citations,
    )
