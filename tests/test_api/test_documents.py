"""
Integration tests for the document lifecycle endpoints.

Mirrors tests/test_api/test_rag.py: the app is driven over ASGITransport with
the corpus store patched out, so what is under test is the HTTP contract
and the auth/scoping gate — not chunking or embedding, which have their own
unit tests.

The scoping tests are regression guards. /documents is a WRITE path into a
user-owned corpus, so it inherits /rag/query's rule exactly: identity wins,
internal service tokens may name a user, anonymous is rejected.

Phase 7 adds the lifecycle contract: every mutation calls `corpus_changed`
(the epoch bump + cache invalidation hook), a no-op dedup ingest does NOT,
and lookups by id return 404 for another tenant's document without confirming
its existence.
"""

import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_optional_user
from app.core.config import settings
from app.main import app
from app.services.corpus import IngestResult

USER_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
OTHER_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")
DOC_ID = uuid.UUID("55555555-5555-5555-5555-555555555555")


async def _auth_user():
    return {"id": str(USER_ID), "username": "tester"}


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def corpus_changed_calls(monkeypatch):
    """Replace the invalidation hook and record which tenant it fired for.

    Autouse: the real hook writes `user_corpus_state` rows and talks to Neo4j,
    neither of which belongs in an HTTP-contract test. Its call/no-call
    behavior IS part of the contract, so the recorder is asserted on below.
    """
    calls: list[uuid.UUID] = []

    async def fake_corpus_changed(db, *, user_id):
        calls.append(user_id)
        return len(calls)

    monkeypatch.setattr("app.api.v1.documents.corpus_changed", fake_corpus_changed)
    return calls


@pytest.fixture
def captured_ingest(monkeypatch):
    """Replace the corpus store's ingest and record the scope it was handed."""
    seen: dict[str, object] = {}

    async def fake_ingest(db, *, user_id, text, title=None, source_uri=None, content_type="text/plain"):
        seen["user_id"] = user_id
        seen["text"] = text
        seen["title"] = title
        return IngestResult(
            document_id=uuid.uuid4(),
            chunk_count=3,
            token_count=42,
            was_deduplicated=False,
            hard_split_chunks=0,
        )

    monkeypatch.setattr("app.api.v1.documents.ingest_document", fake_ingest)
    return seen


async def _post(json_body, headers=None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/v1/documents", json=json_body, headers=headers or {})


async def _get(params=None, headers=None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/api/v1/documents", params=params or {}, headers=headers or {})


async def _delete(document_id, headers=None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.delete(f"/api/v1/documents/{document_id}", headers=headers or {})


async def _put(document_id, json_body, headers=None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.put(
            f"/api/v1/documents/{document_id}", json=json_body, headers=headers or {}
        )


# ── Scoping — regression guards ─────────────────────────────

@pytest.mark.asyncio
async def test_anonymous_ingest_is_rejected(monkeypatch):
    """An anonymous caller must not be able to write into the corpus.

    A document with no owner could never be retrieved by anyone, so accepting
    one would only be an unauthenticated write path with an embedding bill
    attached.
    """

    async def fail_ingest(*args, **kwargs):
        raise AssertionError("anonymous ingest must be rejected before any work")

    monkeypatch.setattr("app.api.v1.documents.ingest_document", fail_ingest)

    response = await _post({"text": "Some document body."})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_anonymous_supplying_user_id_is_forbidden(monkeypatch):
    async def fail_ingest(*args, **kwargs):
        raise AssertionError("must not ingest for a caller who cannot prove that scope")

    monkeypatch.setattr("app.api.v1.documents.ingest_document", fail_ingest)

    response = await _post({"text": "body", "user_id": str(OTHER_ID)})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_authenticated_user_ingests_under_their_own_scope(captured_ingest):
    app.dependency_overrides[get_optional_user] = _auth_user

    response = await _post({"text": "Some document body.", "title": "T"})

    assert response.status_code == 201
    assert captured_ingest["user_id"] == USER_ID
    body = response.json()
    assert body["chunk_count"] == 3
    assert body["deduplicated"] is False


@pytest.mark.asyncio
async def test_authenticated_user_cannot_ingest_for_someone_else(monkeypatch):
    """A body user_id that disagrees with the token is a 403, not a silent
    override in either direction."""
    app.dependency_overrides[get_optional_user] = _auth_user

    async def fail_ingest(*args, **kwargs):
        raise AssertionError("mismatched user_id must be rejected, not honoured")

    monkeypatch.setattr("app.api.v1.documents.ingest_document", fail_ingest)

    response = await _post({"text": "body", "user_id": str(OTHER_ID)})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_service_token_may_name_the_owning_user(captured_ingest, monkeypatch):
    monkeypatch.setattr(
        settings, "RAG_SERVICE_TOKEN", type(settings.RAG_SERVICE_TOKEN)("test-token")
    )

    response = await _post(
        {"text": "body", "user_id": str(OTHER_ID)},
        headers={"X-RAG-Service-Token": "test-token"},
    )

    assert response.status_code == 201
    assert captured_ingest["user_id"] == OTHER_ID


@pytest.mark.asyncio
async def test_unparseable_scope_fails_closed(monkeypatch):
    """A resolved scope that is not a UUID must be rejected, never widened into
    an unscoped write (§5.3 fail closed)."""
    monkeypatch.setattr(
        settings, "RAG_SERVICE_TOKEN", type(settings.RAG_SERVICE_TOKEN)("test-token")
    )

    async def fail_ingest(*args, **kwargs):
        raise AssertionError("must not ingest with an unparseable scope")

    monkeypatch.setattr("app.api.v1.documents.ingest_document", fail_ingest)

    response = await _post(
        {"text": "body", "user_id": "not-a-uuid"},
        headers={"X-RAG-Service-Token": "test-token"},
    )
    assert response.status_code == 400


# ── Request contract ────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_text_is_rejected_by_validation(captured_ingest):
    app.dependency_overrides[get_optional_user] = _auth_user
    response = await _post({"text": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rejected_document_returns_422_not_500(monkeypatch):
    """A document the service refuses (blank, over the token ceiling) is a
    client-input problem, not a server fault."""
    app.dependency_overrides[get_optional_user] = _auth_user

    async def rejecting_ingest(*args, **kwargs):
        raise ValueError("Document produced no chunks")

    monkeypatch.setattr("app.api.v1.documents.ingest_document", rejecting_ingest)

    response = await _post({"text": "   . "})
    assert response.status_code == 422
    assert "no chunks" in response.json()["detail"]


@pytest.mark.asyncio
async def test_dedup_is_reported_to_the_caller(monkeypatch):
    app.dependency_overrides[get_optional_user] = _auth_user

    async def dedup_ingest(*args, **kwargs):
        return IngestResult(
            document_id=uuid.uuid4(),
            chunk_count=5,
            token_count=99,
            was_deduplicated=True,
            hard_split_chunks=2,
        )

    monkeypatch.setattr("app.api.v1.documents.ingest_document", dedup_ingest)

    response = await _post({"text": "body"})
    body = response.json()
    assert body["deduplicated"] is True
    assert body["hard_split_chunks"] == 2


# ── Phase 7: invalidation contract ──────────────────────────

@pytest.mark.asyncio
async def test_ingest_fires_corpus_changed(captured_ingest, corpus_changed_calls):
    """A real write must bump the epoch — invariant 10."""
    app.dependency_overrides[get_optional_user] = _auth_user

    response = await _post({"text": "New content."})

    assert response.status_code == 201
    assert corpus_changed_calls == [USER_ID]


@pytest.mark.asyncio
async def test_dedup_ingest_does_not_fire_corpus_changed(monkeypatch, corpus_changed_calls):
    """A dedup hit wrote nothing. Bumping the epoch anyway would evict the
    tenant's whole cache for a no-op request."""
    app.dependency_overrides[get_optional_user] = _auth_user

    async def dedup_ingest(*args, **kwargs):
        return IngestResult(
            document_id=uuid.uuid4(),
            chunk_count=5,
            token_count=99,
            was_deduplicated=True,
            hard_split_chunks=0,
        )

    monkeypatch.setattr("app.api.v1.documents.ingest_document", dedup_ingest)

    response = await _post({"text": "body"})

    assert response.status_code == 201
    assert corpus_changed_calls == []


# ── Phase 7: list ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_anonymous_list_is_rejected():
    response = await _get()
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_returns_only_the_callers_documents(monkeypatch):
    app.dependency_overrides[get_optional_user] = _auth_user
    seen: dict[str, object] = {}

    class _Doc:
        id = DOC_ID
        title = "T"
        source_uri = None
        content_type = "text/plain"
        status = "completed"
        token_count = 42
        created_at = datetime(2026, 7, 1, tzinfo=UTC)

    async def fake_list(db, *, user_id):
        seen["user_id"] = user_id
        return [(_Doc(), 3)]

    monkeypatch.setattr("app.api.v1.documents.list_documents", fake_list)

    response = await _get()

    assert response.status_code == 200
    assert seen["user_id"] == USER_ID
    documents = response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["document_id"] == str(DOC_ID)
    assert documents[0]["chunk_count"] == 3


# ── Phase 7: delete ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_anonymous_delete_is_rejected(corpus_changed_calls):
    response = await _delete(DOC_ID)
    assert response.status_code == 401
    assert corpus_changed_calls == []


@pytest.mark.asyncio
async def test_delete_fires_corpus_changed(monkeypatch, corpus_changed_calls):
    app.dependency_overrides[get_optional_user] = _auth_user
    seen: dict[str, object] = {}

    async def fake_delete(db, *, user_id, document_id):
        seen["user_id"] = user_id
        seen["document_id"] = document_id
        return True

    monkeypatch.setattr("app.api.v1.documents.delete_document", fake_delete)

    response = await _delete(DOC_ID)

    assert response.status_code == 204
    assert seen == {"user_id": USER_ID, "document_id": DOC_ID}
    assert corpus_changed_calls == [USER_ID]


@pytest.mark.asyncio
async def test_delete_of_unknown_document_is_404_without_invalidation(
    monkeypatch, corpus_changed_calls
):
    """Another tenant's id and a nonexistent id are the same 404 — existence is
    never confirmed across the boundary — and a mutation that didn't happen
    must not evict the tenant's cache."""
    app.dependency_overrides[get_optional_user] = _auth_user

    async def fake_delete(db, *, user_id, document_id):
        return False

    monkeypatch.setattr("app.api.v1.documents.delete_document", fake_delete)

    response = await _delete(DOC_ID)

    assert response.status_code == 404
    assert corpus_changed_calls == []


# ── Phase 7: replace ────────────────────────────────────────

@pytest.mark.asyncio
async def test_anonymous_replace_is_rejected(corpus_changed_calls):
    response = await _put(DOC_ID, {"text": "new body"})
    assert response.status_code == 401
    assert corpus_changed_calls == []


@pytest.mark.asyncio
async def test_replace_fires_corpus_changed(monkeypatch, corpus_changed_calls):
    app.dependency_overrides[get_optional_user] = _auth_user
    seen: dict[str, object] = {}
    new_doc_id = uuid.uuid4()

    async def fake_replace(db, *, user_id, document_id, text, title=None, source_uri=None, content_type="text/plain"):
        seen["user_id"] = user_id
        seen["document_id"] = document_id
        seen["text"] = text
        return IngestResult(
            document_id=new_doc_id,
            chunk_count=2,
            token_count=17,
            was_deduplicated=False,
            hard_split_chunks=0,
        )

    monkeypatch.setattr("app.api.v1.documents.replace_document", fake_replace)

    response = await _put(DOC_ID, {"text": "v2 of the body"})

    assert response.status_code == 200
    assert seen["user_id"] == USER_ID
    assert seen["document_id"] == DOC_ID
    assert response.json()["document_id"] == str(new_doc_id)
    assert corpus_changed_calls == [USER_ID]


@pytest.mark.asyncio
async def test_replace_of_unknown_document_is_404_without_invalidation(
    monkeypatch, corpus_changed_calls
):
    app.dependency_overrides[get_optional_user] = _auth_user

    async def fake_replace(db, **kwargs):
        return None

    monkeypatch.setattr("app.api.v1.documents.replace_document", fake_replace)

    response = await _put(DOC_ID, {"text": "v2"})

    assert response.status_code == 404
    assert corpus_changed_calls == []
