"""
RAG endpoint — Hybrid Graph-Vector retrieval-augmented generation
with semantic intent routing for low-latency simple chat.

Request flow:
  1. classify_intent() — fast LLM call (~200ms) to determine if RAG is needed.
  2a. Simple chat → stream_direct_chat() — direct LLM response (sub-second).
  2b. Knowledge query → SupportCrew kickoff — full CrewAI orchestration.
  3. Optionally queue MemoryCrew as a background task.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import OptionalUser, is_valid_rag_service_request, resolve_rag_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# ── Request / Response Schemas ───────────────────────────────
class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="The user's question.")
    user_id: str | None = Field(default=None, description="Optional user ID for personalized context.")
    stream: bool = Field(default=False, description="If true, stream the response for simple chat.")


class RAGQueryResponse(BaseModel):
    answer: str
    query: str
    user_id: str | None = None
    routed_to: str = Field(description="'cache' for semantic hit, 'direct' for simple chat, 'crewai' for RAG queries.")


# ── Background Memory Processing ────────────────────────────
def _process_memory_background(query: str, answer: str, user_id: str):
    """Fire-and-forget: extract entities from the Q&A and persist to graph."""
    import time

    from agents.crews.memory_crew import MemoryCrew

    transcript = f"User: {query}\nAssistant: {answer}"
    query_snippet = query[:80]

    logger.info("MemoryCrew started — user=%s query='%s'", user_id, query_snippet)
    t0 = time.perf_counter()

    try:
        result = MemoryCrew(user_id=user_id).crew().kickoff(
            inputs={"transcript": transcript}
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "MemoryCrew completed in %.1fs — user=%s query='%s' result_len=%d",
            elapsed, user_id, query_snippet, len(str(result)),
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "MemoryCrew FAILED after %.1fs — user=%s query='%s'",
            elapsed, user_id, query_snippet,
        )


# ── RAG Query Endpoint ───────────────────────────────────────
@router.post(
    "/query",
    status_code=status.HTTP_200_OK,
    summary="Ask a question — auto-routed between direct chat and full RAG",
)
async def rag_query(
    request: RAGQueryRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    user: OptionalUser = None,
):
    """
    Smart-routed query endpoint:

    0. **Cache-Key Normalization** — Whitespace/case folding plus regex PII
       masking, producing the key used against the semantic cache. Not a
       privacy control: the raw query still reaches the LLM and the crew.
    1. **Neo4j Semantic Cache** — A per-user cache; a hit bypasses all
       downstream processing. There is no global/shared cache scope.
    2. **Intent Classification** — A fast LLM call (~200ms) determines if the
       query needs knowledge retrieval or is simple conversation.
    3. **Direct Chat Path** — Greetings, small talk, and opinions bypass
       CrewAI entirely for sub-second responses.
    4. **RAG Path** — Factual and knowledge queries go through the full
       SupportCrew (pgvector + Neo4j + Mem0) pipeline.
    5. **Memory** — Both paths optionally trigger background MemoryCrew and populate graphs.
    """
    from app.services.embeddings import embed_text_async_safe
    from app.services.llm_provider import classify_intent, direct_chat, stream_direct_chat
    from app.services.query_normalizer import normalize_for_cache_key
    from app.services.semantic_cache import get_cached_response, populate_semantic_cache

    resolved_user_id = resolve_rag_user_id(
        request_user_id=request.user_id,
        user=user,
        is_service_request=is_valid_rag_service_request(http_request),
    )

    # Every store this endpoint reads — pgvector messages, the Neo4j
    # subgraph, Mem0 memories — is user-owned, so there is nothing an
    # unscoped caller can legitimately retrieve. This previously fell back to
    # a literal "anonymous" scope, which ran the full crew against every
    # tenant's data. Requiring identity closes that and removes an
    # unauthenticated path to a ~15-LLM-call request.
    if resolved_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication or an internal service token is required for RAG queries.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Step 0: Build the semantic-cache key ────────────────
    # Cache-key normalization only. The raw query is what reaches the
    # classifier, the LLM, and the crew below — this is not a privacy boundary.
    cache_key_query = normalize_for_cache_key(request.query)

    # ── Step 1: Intent Classification ──────────────────────
    needs_rag = await classify_intent(request.query)

    # ── Step 2: User-Scoped Semantic Cache ─────────────────
    query_embedding = None
    if needs_rag and resolved_user_id:
        query_embedding = await embed_text_async_safe(cache_key_query)
        if query_embedding:
            cached_answer = await get_cached_response(
                normalized_query=cache_key_query,
                embedding=query_embedding,
                user_id=resolved_user_id,
            )
            if cached_answer:
                logger.info(
                    "Semantic Cache Hit: user=%s query='%s'",
                    resolved_user_id,
                    request.query[:80],
                )
                return RAGQueryResponse(
                    answer=cached_answer,
                    query=request.query,
                    user_id=resolved_user_id,
                    routed_to="cache",
                )

    # ── Cache Callback Helper ───────────────────────────────
    def _fire_background_callbacks(answer: str, *, populate_cache: bool = False):
        if resolved_user_id:
            background_tasks.add_task(
                _process_memory_background,
                query=request.query,
                answer=answer,
                user_id=resolved_user_id,
            )
        
        if populate_cache and resolved_user_id and query_embedding:
            background_tasks.add_task(
                populate_semantic_cache,
                normalized_query=cache_key_query,
                embedding=query_embedding,
                answer=answer,
                user_id=resolved_user_id,
                session_id=None
            )

    # ── Step 3: Direct Chat (fast path) ────────────────────
    if not needs_rag:
        logger.info("Intent: simple chat — bypassing CrewAI for '%s'", request.query[:80])

        if request.stream:
            # Stream chunks to the client AND collect them so we can cache the full answer afterwards
            collected_chunks: list[str] = []

            async def _stream_generator():
                async for chunk in stream_direct_chat(request.query):
                    collected_chunks.append(chunk)
                    yield chunk
                # After streaming completes, fire cache population + memory
                full_answer = "".join(collected_chunks)
                _fire_background_callbacks(full_answer)

            return StreamingResponse(
                _stream_generator(),
                media_type="text/plain",
            )

        # Non-streaming response
        answer = await direct_chat(request.query)

        _fire_background_callbacks(answer)

        return RAGQueryResponse(
            answer=answer,
            query=request.query,
            user_id=resolved_user_id,
            routed_to="direct",
        )

    # ── Step 4: Full RAG (CrewAI path) ─────────────────────
    import asyncio
    import time

    logger.info("Intent: knowledge query — running SupportCrew for '%s'", request.query[:80])

    from agents.crews.support_crew import SupportCrew

    def _run_crew_sync():
        # user_id is a constructor arg, not a kickoff input: it binds the
        # retrieval tools' scope directly instead of being interpolated into
        # prompt text for the agent to pass along.
        return (
            SupportCrew(user_id=resolved_user_id)
            .crew()
            .kickoff(inputs={"query": request.query})
        )

    t0 = time.perf_counter()
    try:
        # Offload blocking CrewAI execution to thread pool so the
        # FastAPI event loop remains responsive for other requests.
        result = await asyncio.to_thread(_run_crew_sync)
        elapsed = time.perf_counter() - t0
        answer = str(result).strip()

        logger.info(
            "SupportCrew completed in %.1fs — query='%s' answer_len=%d",
            elapsed, request.query[:80], len(answer),
        )

        # Validate crew output — if empty or suspiciously short, fall back to direct LLM
        if not answer or len(answer) < 5:
            logger.warning(
                "SupportCrew returned empty/invalid result, falling back to direct LLM"
            )
            answer = await direct_chat(request.query)
            routed = "direct_fallback"
        else:
            routed = "crewai"

    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "SupportCrew FAILED after %.1fs — query='%s'", elapsed, request.query[:80]
        )
        # Graceful degradation: fall back to direct LLM instead of 500
        answer = await direct_chat(request.query)
        routed = "direct_fallback"

    _fire_background_callbacks(answer, populate_cache=(routed == "crewai"))

    return RAGQueryResponse(
        answer=answer,
        query=request.query,
        user_id=resolved_user_id,
        routed_to=routed,
    )
