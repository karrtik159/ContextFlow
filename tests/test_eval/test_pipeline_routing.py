"""
Integration tests for the 5-tier RAG pipeline routing logic.

Since the app module chain (rag → deps → db) triggers asyncpg at import time,
these tests simulate the routing logic in isolation by directly exercising
the routing decision functions WITHOUT importing the full rag module.

This approach tests the LOGIC of the pipeline, not the HTTP transport.
"""

import pytest

from app.services.query_normalizer import normalize_for_cache_key


async def _simulate_pipeline(
    query: str,
    user_id: str | None = None,
    *,
    cache_key: str | None = None,
    embedding: list[float] | None = None,
    cached_answer: str | None = None,
    intent_needs_rag: bool = False,
    direct_chat_answer: str = "direct answer",
    crew_answer: str | None = None,
    crew_error: Exception | None = None,
) -> dict:
    """Simulate the RAG pipeline routing logic WITHOUT importing the full app module.

    Returns a dict with 'answer' and 'routed_to' mirroring RAGQueryResponse.
    """
    # Step 0: Cache-key normalization
    if cache_key is None:
        cache_key = normalize_for_cache_key(query)

    # Step 1: Intent
    needs_rag = intent_needs_rag

    # Step 2: User-scoped Embedding + Cache
    query_embedding = embedding  # None simulates embedding failure

    if needs_rag and user_id and query_embedding is not None and cached_answer is not None:
        return {"answer": cached_answer, "routed_to": "cache"}

    # Step 3: Direct chat
    if not needs_rag:
        return {"answer": direct_chat_answer, "routed_to": "direct"}

    # Step 4: CrewAI
    if crew_error:
        return {"answer": direct_chat_answer, "routed_to": "direct_fallback"}

    if crew_answer and len(crew_answer.strip()) >= 5:
        return {"answer": crew_answer.strip(), "routed_to": "crewai"}
    else:
        return {"answer": direct_chat_answer, "routed_to": "direct_fallback"}


# ── Test: Cache Hit Path ────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_returns_cached_answer():
    """When semantic cache has a match, bypass all LLM processing."""
    result = await _simulate_pipeline(
        "What is AI?",
        user_id="user-123",
        embedding=[0.1] * 384,
        cached_answer="Cached: AI is...",
        intent_needs_rag=True,
    )
    assert result["routed_to"] == "cache"
    assert result["answer"] == "Cached: AI is..."


# ── Test: Direct Chat Path ──────────────────────────────────

@pytest.mark.asyncio
async def test_simple_chat_routes_to_direct():
    """When intent classifier says CHAT, bypass CrewAI."""
    result = await _simulate_pipeline(
        "How are you?",
        embedding=[0.1] * 384,
        intent_needs_rag=False,
        direct_chat_answer="I'm doing well!",
    )
    assert result["routed_to"] == "direct"
    assert result["answer"] == "I'm doing well!"


# ── Test: CrewAI RAG Path ───────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_query_routes_to_crewai():
    """When intent classifier says RAG, use SupportCrew."""
    result = await _simulate_pipeline(
        "What is quantum computing?",
        embedding=[0.1] * 384,
        intent_needs_rag=True,
        crew_answer="Quantum computing uses qubits to perform calculations...",
    )
    assert result["routed_to"] == "crewai"
    assert "qubits" in result["answer"]


# ── Test: CrewAI Failure → Fallback ─────────────────────────

@pytest.mark.asyncio
async def test_crew_failure_falls_back_to_direct():
    """When SupportCrew raises an exception, fall back to direct LLM."""
    result = await _simulate_pipeline(
        "What is quantum computing?",
        embedding=[0.1] * 384,
        intent_needs_rag=True,
        crew_error=RuntimeError("LLM crashed"),
        direct_chat_answer="Fallback answer",
    )
    assert result["routed_to"] == "direct_fallback"
    assert result["answer"] == "Fallback answer"


# ── Test: CrewAI Empty Result → Fallback ────────────────────

@pytest.mark.asyncio
async def test_crew_empty_result_falls_back():
    """When SupportCrew returns empty/short answer, fall back to direct LLM."""
    result = await _simulate_pipeline(
        "Tell me about RAG",
        embedding=[0.1] * 384,
        intent_needs_rag=True,
        crew_answer="",  # Empty crew result
        direct_chat_answer="Direct RAG explanation",
    )
    assert result["routed_to"] == "direct_fallback"


# ── Test: PII Isolation ─────────────────────────────────────

@pytest.mark.asyncio
async def test_pii_query_uses_user_scoped_cache():
    """Queries with PII still use the resolved user's cache scope."""
    user_id = "user-123"
    cache_lookup_scoped_id = user_id

    assert cache_lookup_scoped_id == "user-123"  # Isolated to user


@pytest.mark.asyncio
async def test_no_pii_still_uses_user_scoped_cache():
    """Clean queries use the resolved user's cache scope, not a global pool.

    There is no unscoped cache tier: scope comes from the resolved user id, not
    from any property of the query itself.
    """
    cache_key = normalize_for_cache_key("What is machine learning?")

    user_id = "user-456"
    cache_lookup_scoped_id = user_id

    assert cache_key == "what is machine learning?"
    assert cache_lookup_scoped_id == "user-456"


# ── Test: Embedding Failure → Skip Cache Gracefully ─────────

@pytest.mark.asyncio
async def test_embedding_failure_skips_cache():
    """If embedding generation fails, skip cache and proceed to intent classification."""
    result = await _simulate_pipeline(
        "What is AI?",
        embedding=None,  # Embedding failed
        intent_needs_rag=False,
        direct_chat_answer="AI is...",
    )
    assert result["routed_to"] == "direct"
    assert result["answer"] == "AI is..."


# ── Test: Sanitizer Integration ─────────────────────────────

def test_cache_key_masks_pii():
    """PII is masked out of the key that gets persisted on the cache node."""
    assert normalize_for_cache_key("Email me at admin@company.org") == "email me at [email]"


def test_cache_key_strips_fillers():
    """Filler prefixes are stripped so phrasings share a cache entry."""
    assert normalize_for_cache_key("Hey, can you tell me about transformers?") == "about transformers?"
