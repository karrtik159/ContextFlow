"""
Chunker registry + chunk provenance (Phase 8).

The registry's contract: known text types resolve to the CALLER's structural
default silently; unknown types resolve to it LOUDLY; a registered type gets
its own chunker. And ingestion must stamp every chunk row with the chunker
version and the hard-split flag (invariant 11) — the provenance that makes
scripts/rechunk_corpus.py a targeted operation instead of a full-corpus guess.
"""

from __future__ import annotations

import logging
import uuid

import pytest

from app.models.document import Chunk, Document
from app.services.chunkers import _REGISTRY, register_chunker, resolve_chunker
from app.services.chunking import CHUNKER_VERSION, ChunkDraft
from app.services.corpus import store

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _default():
    raise AssertionError("must not be called")


# ── resolve_chunker ─────────────────────────────────────────


def test_known_text_types_resolve_to_the_callers_default(caplog):
    with caplog.at_level(logging.WARNING, logger="app.services.chunkers"):
        for content_type in ("text/plain", "text/markdown", "text/x-markdown"):
            assert resolve_chunker(content_type, default=_default) is _default
    assert caplog.records == [], "known types must resolve silently"


def test_content_type_parameters_and_case_are_normalized(caplog):
    with caplog.at_level(logging.WARNING, logger="app.services.chunkers"):
        resolved = resolve_chunker("TEXT/Markdown; charset=utf-8", default=_default)
    assert resolved is _default
    assert caplog.records == []


def test_unknown_type_falls_back_loudly(caplog):
    with caplog.at_level(logging.WARNING, logger="app.services.chunkers"):
        resolved = resolve_chunker("application/pdf", default=_default)
    assert resolved is _default, "unknown types degrade to the structural default"
    assert any("application/pdf" in r.getMessage() for r in caplog.records), (
        "the degradation must be observable — never quietly"
    )


def test_registered_chunker_wins(monkeypatch):
    def html_chunker(*args, **kwargs):
        return []

    # Isolate the registry mutation to this test.
    monkeypatch.setitem(_REGISTRY, "text/html", html_chunker)
    assert resolve_chunker("text/html", default=_default) is html_chunker


def test_register_chunker_rejects_empty_type():
    with pytest.raises(ValueError, match="content type"):
        register_chunker("", lambda: None)


# ── Ingestion stamps provenance ─────────────────────────────


class FakeSession:
    def __init__(self):
        self.added: list[object] = []

    async def scalar(self, _stmt):
        return None  # no existing document — never a dedup hit

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    async def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()


@pytest.mark.asyncio
async def test_ingest_stamps_chunker_version_and_hard_split(monkeypatch):
    drafts = [
        ChunkDraft(text="clean chunk", heading_path=None, char_start=0, char_end=11, token_count=3),
        ChunkDraft(
            text="mangled chunk", heading_path=None, char_start=11, char_end=24,
            token_count=3, was_hard_split=True,
        ),
    ]

    async def fake_embed(texts):
        return [[0.0] for _ in texts]

    monkeypatch.setattr(store, "chunk_document", lambda *a, **k: drafts)
    monkeypatch.setattr(store, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(store, "embed_texts_async", fake_embed)

    db = FakeSession()
    await store.ingest_document(db, user_id=USER, text="clean chunk mangled chunk")

    chunks = [obj for obj in db.added if isinstance(obj, Chunk)]
    assert len(chunks) == 2
    assert all(c.chunker_version == CHUNKER_VERSION for c in chunks), (
        "every chunk row must record the chunker that produced it (invariant 11)"
    )
    assert [c.was_hard_split for c in chunks] == [False, True], (
        "the hard-split quality signal must be persisted, not just logged"
    )


@pytest.mark.asyncio
async def test_ingest_keeps_the_module_binding_as_the_chunker_seam(monkeypatch):
    """resolve_chunker must hand back store.chunk_document — the patched
    binding — for default text types, not a private import of the real one."""
    calls: list[str] = []

    def recording_chunker(text, **kwargs):
        calls.append(text)
        return [
            ChunkDraft(text=text, heading_path=None, char_start=0, char_end=len(text), token_count=1)
        ]

    async def fake_embed(texts):
        return [[0.0] for _ in texts]

    monkeypatch.setattr(store, "chunk_document", recording_chunker)
    monkeypatch.setattr(store, "count_tokens", lambda text: 1)
    monkeypatch.setattr(store, "embed_texts_async", fake_embed)

    db = FakeSession()
    await store.ingest_document(db, user_id=USER, text="body", content_type="text/markdown")

    assert calls == ["body"], "the store's own chunk_document binding must be used"
    documents = [obj for obj in db.added if isinstance(obj, Document)]
    assert documents and documents[0].raw_text == "body"
