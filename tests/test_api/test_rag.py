"""
Integration test for the RAG query endpoint.

NOTE: This test requires the full app stack (PostgreSQL, asyncpg) and runs
in Docker/CI environments. It will NOT work in local development without
the database running.

The test mocks the SupportCrew, intent classifier, and embedding service
to isolate the HTTP transport + routing logic.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_optional_user
from app.main import app


async def _needs_rag(*args, **kwargs):
    return True


async def _simple_chat(*args, **kwargs):
    return False


async def _no_embedding(*args, **kwargs):
    return None


async def _direct_answer(*args, **kwargs):
    return "Hey there!"


class _FakeCrewRunner:
    def kickoff(self, *, inputs):
        assert inputs["query"] == "How does the worker reach the backend?"
        # user_id is a SupportCrew constructor arg now, never a kickoff input.
        assert "user_id" not in inputs
        return "The worker calls the FastAPI RAG endpoint over HTTP."


class _FakeSupportCrew:
    """Records the tenant scope the endpoint constructed the crew with."""

    last_user_id: str | None = None

    def __init__(self, user_id):
        type(self).last_user_id = user_id

    def crew(self):
        return _FakeCrewRunner()


async def _auth_user():
    return {"id": "user-123", "username": "user-123"}


# ── Phase 2 helpers ──────────────────────────────────────────
# Retrieval is deterministic now, so the crew path needs a real embedding and a
# stubbed pipeline rather than the old "no embedding" shortcut. A None
# embedding legitimately means dense retrieval cannot run, which degrades to
# direct_fallback (see test_rag_query_degrades_when_embedding_unavailable).


async def _an_embedding(*args, **kwargs):
    return [0.1, 0.2, 0.3]


def _stub_retrieval(monkeypatch, chunk_text="Retrieved supporting evidence."):
    """Make run_retrieval return one chunk without touching PostgreSQL."""
    import uuid as _uuid

    from app.services.retrieval.contracts import RetrievalTrace, RetrievedChunk

    async def fake_run_retrieval(db, **kwargs):
        trace = RetrievalTrace(
            trace_id=_uuid.uuid4(),
            user_id=kwargs["user_id"],
            original_query=kwargs["original_query"],
            normalized_query=kwargs["retrieval_query"],
        )
        trace.final_chunks = [
            RetrievedChunk(
                text=chunk_text, source="vector", rank=1, score=0.9,
                chunk_id=_uuid.uuid4(), citation_label="[1]",
            )
        ]
        return trace

    async def fake_persist_trace(**kwargs):
        return None

    async def fake_populate_cache(**kwargs):
        return None

    monkeypatch.setattr("app.services.rag_service.run_retrieval", fake_run_retrieval)
    monkeypatch.setattr("app.services.trace_store.persist_trace", fake_persist_trace)
    # A grounded crew answer is now genuinely cacheable, so the background task
    # fires and would reach Neo4j.
    monkeypatch.setattr(
        "app.services.semantic_cache.populate_semantic_cache", fake_populate_cache
    )


@pytest.mark.asyncio
async def test_rag_query_crewai_path(monkeypatch):
    """Full HTTP roundtrip through the RAG endpoint — crew path."""
    app.dependency_overrides[get_optional_user] = _auth_user
    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", _FakeSupportCrew)
    # Force intent classifier to say "needs RAG"
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )
    monkeypatch.setattr(
        "app.services.embeddings.embed_text_async_safe",
        _an_embedding,
    )

    async def _no_cache(**kwargs):
        return None

    monkeypatch.setattr("app.services.semantic_cache.get_cached_response", _no_cache)
    _stub_retrieval(monkeypatch)
    monkeypatch.setattr("app.api.v1.rag._process_memory_background", lambda *args, **kwargs: None)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"query": "How does the worker reach the backend?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The worker calls the FastAPI RAG endpoint over HTTP."
    assert data["query"] == "How does the worker reach the backend?"
    assert data["user_id"] == "user-123"
    assert data["routed_to"] == "crewai"
    # The crew must be bound to the authenticated identity, not a sentinel.
    assert _FakeSupportCrew.last_user_id == "user-123"


@pytest.mark.asyncio
async def test_rag_query_direct_path(monkeypatch):
    """Full HTTP roundtrip — direct chat path (bypasses CrewAI)."""
    app.dependency_overrides[get_optional_user] = _auth_user
    # Force intent classifier to say "simple chat"
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _simple_chat,
    )
    # Mock direct_chat
    monkeypatch.setattr(
        "app.services.llm_provider.direct_chat",
        _direct_answer,
    )
    # Skip embedding + cache
    monkeypatch.setattr(
        "app.services.embeddings.embed_text_async_safe",
        _no_embedding,
    )
    monkeypatch.setattr("app.api.v1.rag._process_memory_background", lambda *args, **kwargs: None)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"query": "Hello!"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hey there!"
    assert data["routed_to"] == "direct"


@pytest.mark.asyncio
async def test_rag_query_rejects_spoofed_user_id(monkeypatch):
    app.dependency_overrides[get_optional_user] = _auth_user
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"query": "What did I decide?", "user_id": "other-user"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_rag_query_service_token_can_supply_user_id(monkeypatch):
    monkeypatch.setattr(
        "app.api.deps.settings.RAG_SERVICE_TOKEN",
        type("Secret", (), {"get_secret_value": lambda self: "service-secret"})(),
    )
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )
    monkeypatch.setattr(
        "app.services.embeddings.embed_text_async_safe",
        _an_embedding,
    )

    async def _no_cache(**kwargs):
        return None

    monkeypatch.setattr("app.services.semantic_cache.get_cached_response", _no_cache)
    _stub_retrieval(monkeypatch)
    monkeypatch.setattr("app.api.v1.rag._process_memory_background", lambda *args, **kwargs: None)

    captured_scope: dict[str, str] = {}

    class FakeCrewRunner:
        def kickoff(self, *, inputs):
            assert "user_id" not in inputs
            return "Scoped answer from the knowledge service."

    class FakeSupportCrew:
        def __init__(self, user_id):
            captured_scope["user_id"] = user_id

        def crew(self):
            return FakeCrewRunner()

    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", FakeSupportCrew)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/query",
            headers={"X-RAG-Service-Token": "service-secret"},
            json={"query": "What did I decide?", "user_id": "user-123"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user-123"
    assert data["routed_to"] == "crewai"
    # Service-token callers may assert a scope; it must reach the crew intact.
    assert captured_scope["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_rag_query_anonymous_is_rejected(monkeypatch):
    """Anonymous callers get 401 — they have no retrievable, scoped corpus.

    Every store behind this endpoint is user-owned, so an unscoped run could
    only either return nothing or (as it previously did) read across all
    tenants. It also removes an unauthenticated path to a multi-agent,
    ~15-LLM-call request.
    """

    def fail_crew(*args, **kwargs):
        raise AssertionError("anonymous RAG requests must never construct a crew")

    async def fail_embedding(*args, **kwargs):
        raise AssertionError("anonymous RAG requests must not generate cache embeddings")

    async def fail_classify(*args, **kwargs):
        raise AssertionError("anonymous requests must be rejected before intent classification")

    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", fail_crew)
    monkeypatch.setattr("app.services.llm_provider.classify_intent", fail_classify)
    monkeypatch.setattr("app.services.embeddings.embed_text_async_safe", fail_embedding)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "How does the worker reach the backend?"},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_rag_query_authenticated_cache_is_user_scoped(monkeypatch):
    captured: dict[str, object] = {}
    app.dependency_overrides[get_optional_user] = _auth_user

    async def embedding(*args, **kwargs):
        return [0.1, 0.2, 0.3]

    async def cached_response(*, normalized_query, embedding, user_id):
        captured["normalized_query"] = normalized_query
        captured["embedding"] = embedding
        captured["user_id"] = user_id
        return "Cached scoped answer"

    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )
    monkeypatch.setattr(
        "app.services.embeddings.embed_text_async_safe",
        embedding,
    )
    monkeypatch.setattr(
        "app.services.semantic_cache.get_cached_response",
        cached_response,
    )

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"query": "What did I decide?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Cached scoped answer"
    assert data["user_id"] == "user-123"
    assert data["routed_to"] == "cache"
    assert captured["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_rag_query_degrades_when_embedding_unavailable(monkeypatch):
    """No embedding means the dense arm — the primary one — cannot run.

    Answering anyway from graph and memory scraps would produce a thinly
    grounded answer wearing the "crewai" label. Degrading to direct chat and
    saying so via routed_to is the honest outcome.
    """
    app.dependency_overrides[get_optional_user] = _auth_user
    monkeypatch.setattr("app.services.llm_provider.classify_intent", _needs_rag)
    monkeypatch.setattr("app.services.embeddings.embed_text_async_safe", _no_embedding)
    monkeypatch.setattr("app.services.llm_provider.direct_chat", _direct_answer)

    def fail_crew(*args, **kwargs):
        raise AssertionError("must not run the crew without a usable embedding")

    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", fail_crew)
    monkeypatch.setattr("app.api.v1.rag._process_memory_background", lambda *a, **k: None)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query", json={"query": "What did I decide?"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["routed_to"] == "direct_fallback"


@pytest.mark.asyncio
async def test_rag_query_returns_citations_and_trace_id(monkeypatch):
    """Every non-cached knowledge answer carries resolvable citations."""
    app.dependency_overrides[get_optional_user] = _auth_user
    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", _FakeSupportCrew)
    monkeypatch.setattr("app.services.llm_provider.classify_intent", _needs_rag)
    monkeypatch.setattr("app.services.embeddings.embed_text_async_safe", _an_embedding)

    async def _no_cache(**kwargs):
        return None

    monkeypatch.setattr("app.services.semantic_cache.get_cached_response", _no_cache)
    _stub_retrieval(monkeypatch)
    monkeypatch.setattr("app.api.v1.rag._process_memory_background", lambda *a, **k: None)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"query": "How does the worker reach the backend?"},
            )
    finally:
        app.dependency_overrides.clear()

    data = response.json()
    assert data["trace_id"], "a knowledge answer must carry a trace id"
    assert len(data["citations"]) == 1
    citation = data["citations"][0]
    assert citation["label"] == "[1]"
    assert citation["source"] == "vector"
    assert citation["chunk_id"]
