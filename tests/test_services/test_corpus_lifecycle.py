"""
Unit tests for the corpus lifecycle component (Phase 7).

No database and no Neo4j: sessions are fakes, the invalidation hook is patched.
Under test are the contracts —

- store: tenant scope is required everywhere; delete/replace make "not yours"
  and "not there" indistinguishable; replace deletes before ingesting so
  identical content rebuilds rather than dedup-matching the doomed row.
- events: `corpus_changed` bumps the epoch first and treats the Neo4j delete
  as best-effort; `get_corpus_epoch_safe` never raises and never poisons the
  caller's transaction (it degrades to epoch 0).
"""

from __future__ import annotations

import uuid

import pytest

from app.services.corpus import events, store

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
DOC_ID = uuid.UUID("99999999-9999-9999-9999-999999999999")


# ── Fakes ───────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class _FakeNested:
    """Stand-in for AsyncSession.begin_nested() — a passthrough async CM."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False  # propagate, like a real savepoint


class _FakeDb:
    def __init__(self, *, scalar_value=None, execute_value=None, scalar_raises=None):
        self._scalar_value = scalar_value
        self._execute_value = execute_value
        self._scalar_raises = scalar_raises
        self.deleted: list[object] = []
        self.statements: list[object] = []

    async def scalar(self, stmt):
        if self._scalar_raises is not None:
            raise self._scalar_raises
        self.statements.append(stmt)
        return self._scalar_value

    async def execute(self, stmt):
        self.statements.append(stmt)
        return _FakeResult(self._execute_value)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def flush(self):
        pass

    def begin_nested(self):
        return _FakeNested()


# ── Store: tenant scope is structural ───────────────────────


@pytest.mark.asyncio
async def test_list_documents_requires_user_id():
    with pytest.raises(ValueError, match="user_id"):
        await store.list_documents(_FakeDb(), user_id=None)


@pytest.mark.asyncio
async def test_delete_document_requires_user_id():
    with pytest.raises(ValueError, match="user_id"):
        await store.delete_document(_FakeDb(), user_id=None, document_id=DOC_ID)


@pytest.mark.asyncio
async def test_replace_document_requires_user_id():
    with pytest.raises(ValueError, match="user_id"):
        await store.replace_document(_FakeDb(), user_id=None, document_id=DOC_ID, text="x")


# ── Store: delete / replace semantics ───────────────────────


@pytest.mark.asyncio
async def test_delete_returns_false_when_not_found():
    """Covers both "not there" and "not yours" — the scoped SELECT makes them
    the same case, which is what lets the API 404 without leaking existence."""
    db = _FakeDb(scalar_value=None)
    assert await store.delete_document(db, user_id=USER, document_id=DOC_ID) is False
    assert db.deleted == []


@pytest.mark.asyncio
async def test_delete_removes_the_owned_document():
    sentinel = object()
    db = _FakeDb(scalar_value=sentinel)
    assert await store.delete_document(db, user_id=USER, document_id=DOC_ID) is True
    assert db.deleted == [sentinel]


@pytest.mark.asyncio
async def test_replace_returns_none_when_not_found():
    db = _FakeDb(scalar_value=None)
    result = await store.replace_document(db, user_id=USER, document_id=DOC_ID, text="v2")
    assert result is None
    assert db.deleted == []


@pytest.mark.asyncio
async def test_replace_deletes_old_document_then_ingests(monkeypatch):
    """Order matters: the delete must flush before ingest runs, so identical
    content rebuilds a fresh document instead of dedup-matching the row that
    is being replaced."""
    old_doc = object()
    db = _FakeDb(scalar_value=old_doc)
    order: list[str] = []

    db_delete = db.delete

    async def tracking_delete(obj):
        order.append("delete")
        await db_delete(obj)

    db.delete = tracking_delete

    async def fake_ingest(db_, *, user_id, text, title=None, source_uri=None, content_type="text/plain"):
        order.append("ingest")
        assert user_id == USER
        assert text == "v2 body"
        return store.IngestResult(
            document_id=uuid.uuid4(),
            chunk_count=1,
            token_count=2,
            was_deduplicated=False,
            hard_split_chunks=0,
        )

    monkeypatch.setattr(store, "ingest_document", fake_ingest)

    result = await store.replace_document(db, user_id=USER, document_id=DOC_ID, text="v2 body")

    assert result is not None
    assert order == ["delete", "ingest"]
    assert db.deleted == [old_doc]


# ── Events: epoch + invalidation ────────────────────────────


@pytest.mark.asyncio
async def test_corpus_changed_bumps_epoch_and_invalidates(monkeypatch):
    db = _FakeDb(execute_value=7)
    invalidated: list[str] = []

    async def fake_invalidate(user_id):
        invalidated.append(user_id)

    monkeypatch.setattr(
        "app.services.semantic_cache.invalidate_user_cache", fake_invalidate
    )

    epoch = await events.corpus_changed(db, user_id=USER)

    assert epoch == 7
    assert invalidated == [str(USER)]


@pytest.mark.asyncio
async def test_corpus_changed_survives_invalidation_failure(monkeypatch):
    """The Neo4j delete is hygiene. Its failure must not fail the mutation —
    the epoch bump already guarantees the stale entries cannot be served."""
    db = _FakeDb(execute_value=3)

    async def broken_invalidate(user_id):
        raise ConnectionError("neo4j is down")

    monkeypatch.setattr(
        "app.services.semantic_cache.invalidate_user_cache", broken_invalidate
    )

    epoch = await events.corpus_changed(db, user_id=USER)
    assert epoch == 3


@pytest.mark.asyncio
async def test_get_corpus_epoch_defaults_to_zero():
    db = _FakeDb(scalar_value=None)
    assert await events.get_corpus_epoch(db, user_id=USER) == 0


@pytest.mark.asyncio
async def test_get_corpus_epoch_safe_rejects_non_uuid_scope():
    """Internal service tokens may name arbitrary ids; a non-UUID scope has no
    corpus and must read as epoch 0 without touching the database."""
    db = _FakeDb(scalar_raises=AssertionError("must not query for a non-UUID scope"))
    assert await events.get_corpus_epoch_safe(db, "voice-service") == 0


@pytest.mark.asyncio
async def test_get_corpus_epoch_safe_degrades_on_db_error():
    """A statement error (unmigrated table) must degrade to epoch 0 inside a
    savepoint, not fail the request or poison the caller's transaction."""
    db = _FakeDb(scalar_raises=RuntimeError("relation user_corpus_state does not exist"))
    assert await events.get_corpus_epoch_safe(db, str(USER)) == 0
