"""
Chat message write-path tests.

Phase 9 changed the write path: a message's tenant scope is the SESSION's
owner, resolved server-side and denormalized onto the row, and the embedding
input is bounded to the model's token ceiling rather than silently truncated.
The db fake therefore has to answer `db.get(ChatSession, ...)`.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.v1.chat import create_message
from app.schemas.message import MessageCreate

OWNER_ID = uuid4()


class _FakeDb:
    """Minimal AsyncSession surface create_message touches: get()."""

    def __init__(self, session):
        self._session = session

    async def get(self, model, pk):
        return self._session


def _session(user_id=OWNER_ID):
    return SimpleNamespace(id=uuid4(), user_id=user_id)


@pytest.mark.asyncio
async def test_create_message_embeds_user_message(monkeypatch):
    captured = {}

    async def fake_embed_text_async_safe(content):
        captured["embedded_content"] = content
        return [0.1, 0.2, 0.3]

    async def fake_create(db, object, schema_to_select, return_as_model):
        captured["object"] = object
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr("app.api.v1.chat.embed_text_async_safe", fake_embed_text_async_safe)
    monkeypatch.setattr("app.api.v1.chat.message_crud.create", fake_create)

    session_id = uuid4()
    await create_message(
        session_id,
        MessageCreate(role="user", content="Remember that I prefer concise deployment checklists."),
        db=_FakeDb(_session()),
    )

    assert captured["embedded_content"] == "Remember that I prefer concise deployment checklists."
    assert captured["object"].session_id == session_id
    assert captured["object"].embedding == [0.1, 0.2, 0.3]
    # Phase 9: the row is stamped with the session's owner, server-side.
    assert captured["object"].user_id == OWNER_ID
    assert captured["object"].embedding_truncated is False


@pytest.mark.asyncio
async def test_create_message_does_not_embed_system_message(monkeypatch):
    captured = {}

    async def fail_embed_text_async_safe(content):
        raise AssertionError("system messages should not be embedded")

    async def fake_create(db, object, schema_to_select, return_as_model):
        captured["object"] = object
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr("app.api.v1.chat.embed_text_async_safe", fail_embed_text_async_safe)
    monkeypatch.setattr("app.api.v1.chat.message_crud.create", fake_create)

    await create_message(
        uuid4(),
        MessageCreate(role="system", content="Internal instruction."),
        db=_FakeDb(_session()),
    )

    assert captured["object"].embedding is None
    assert captured["object"].user_id == OWNER_ID


@pytest.mark.asyncio
async def test_create_message_404s_for_unknown_session(monkeypatch):
    from fastapi import HTTPException

    async def fake_create(*a, **k):
        raise AssertionError("must not write a message for a session that does not exist")

    monkeypatch.setattr("app.api.v1.chat.message_crud.create", fake_create)

    with pytest.raises(HTTPException) as excinfo:
        await create_message(
            uuid4(),
            MessageCreate(role="user", content="hi"),
            db=_FakeDb(None),
        )
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_create_message_budgets_an_oversized_body(monkeypatch):
    """A message over the embedding ceiling is embedded from a head and flagged,
    not silently truncated by the encoder (invariant 6)."""
    from app.core.config import settings

    captured = {}

    async def fake_embed_text_async_safe(content):
        captured["embedded_content"] = content
        return [0.1]

    async def fake_create(db, object, schema_to_select, return_as_model):
        captured["object"] = object
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr("app.api.v1.chat.embed_text_async_safe", fake_embed_text_async_safe)
    monkeypatch.setattr("app.api.v1.chat.message_crud.create", fake_create)
    # A tiny ceiling forces the budget path without needing a huge string.
    monkeypatch.setattr(settings, "EMBEDDING_MAX_TOKENS", 3)

    long_body = "word " * 200
    await create_message(
        uuid4(), MessageCreate(role="user", content=long_body), db=_FakeDb(_session())
    )

    assert captured["object"].embedding_truncated is True
    assert len(captured["embedded_content"]) < len(long_body)
    # What was embedded is the head that fit, not the whole body.
    assert captured["embedded_content"] == long_body[: len(captured["embedded_content"])]
