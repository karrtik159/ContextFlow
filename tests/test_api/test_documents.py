"""
Integration tests for the document ingestion endpoint.

Mirrors tests/test_api/test_rag.py: the app is driven over ASGITransport with
the ingestion service patched out, so what is under test is the HTTP contract
and the auth/scoping gate — not chunking or embedding, which have their own
unit tests.

The scoping tests are regression guards. /documents is a WRITE path into a
user-owned corpus, so it inherits /rag/query's rule exactly: identity wins,
internal service tokens may name a user, anonymous is rejected.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_optional_user
from app.core.config import settings
from app.main import app
from app.services.ingestion import IngestResult

USER_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
OTHER_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")


async def _auth_user():
    return {"id": str(USER_ID), "username": "tester"}


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def captured_ingest(monkeypatch):
    """Replace the ingestion service and record the scope it was handed."""
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
