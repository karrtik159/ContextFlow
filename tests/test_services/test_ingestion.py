"""
Unit tests for document ingestion.

No database and no embedding model: the session is a fake and the embedding /
tokenizer calls are patched. What is under test here is the ingestion contract —
tenant propagation, idempotency, batching, and the refusal to store a chunk the
encoder would silently truncate — not SQLAlchemy or OpenAI.

The tenant-scoping tests are regression guards for
docs/ADVANCED_RAG_PLAN.md §5 invariants 1-3. If one fails, the change is wrong.
"""

from __future__ import annotations

import uuid

import pytest

from app.models.document import Chunk, Document
from app.services import ingestion
from app.services.chunking import ChunkDraft
from app.services.ingestion import compute_checksum, ingest_document

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_USER = uuid.UUID("22222222-2222-2222-2222-222222222222")


class FakeScalars(list):
    pass


class FakeSession:
    """Minimal AsyncSession stand-in covering exactly what ingestion uses."""

    def __init__(self, existing_document: Document | None = None):
        self.added: list[object] = []
        self._existing = existing_document
        self.flush_count = 0

    async def scalar(self, _stmt):
        return self._existing

    async def scalars(self, _stmt):
        return FakeScalars()

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    async def flush(self):
        self.flush_count += 1
        # Mimic the DB assigning defaults on flush.
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    @property
    def documents(self) -> list[Document]:
        return [o for o in self.added if isinstance(o, Document)]

    @property
    def chunks(self) -> list[Chunk]:
        return [o for o in self.added if isinstance(o, Chunk)]


@pytest.fixture
def patched(monkeypatch):
    """Deterministic tokenizer and a recording batch-embedder."""
    calls: dict[str, list] = {"embed": []}

    def fake_count_tokens(text: str) -> int:
        return max(len(text.split()), len(text) // 4)

    async def fake_embed_texts_async(texts: list[str]) -> list[list[float]]:
        calls["embed"].append(list(texts))
        return [[float(i)] * 3 for i in range(len(texts))]

    monkeypatch.setattr(ingestion, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(ingestion, "embed_texts_async", fake_embed_texts_async)
    return calls


# ── Rejection paths ─────────────────────────────────────────

@pytest.mark.parametrize("blank", ["", "   ", "\n\t\n"])
async def test_blank_document_is_rejected(patched, blank):
    db = FakeSession()
    with pytest.raises(ValueError, match="empty document"):
        await ingest_document(db, user_id=USER, text=blank)
    assert db.added == [], "nothing may be persisted for a rejected document"


async def test_chunk_over_the_model_ceiling_aborts_the_ingest(patched, monkeypatch):
    """A chunk past EMBEDDING_MAX_TOKENS must fail the whole ingest.

    MiniLM truncates such input with no warning, so a stored row would carry an
    embedding representing only its first fragment — undetectable downstream.
    """
    monkeypatch.setattr(
        ingestion,
        "chunk_document",
        lambda *a, **k: [
            ChunkDraft(
                text="x" * 100,
                heading_path=None,
                char_start=0,
                char_end=100,
                token_count=999_999,
            )
        ],
    )
    db = FakeSession()
    with pytest.raises(ValueError, match="over the .* ceiling"):
        await ingest_document(db, user_id=USER, text="some text")
    assert db.chunks == [], "no chunk may be persisted when one is over the ceiling"


async def test_document_producing_no_chunks_is_rejected(patched, monkeypatch):
    monkeypatch.setattr(ingestion, "chunk_document", lambda *a, **k: [])
    db = FakeSession()
    with pytest.raises(ValueError, match="no chunks"):
        await ingest_document(db, user_id=USER, text="text that chunks to nothing")


async def test_vector_count_mismatch_is_fatal(patched, monkeypatch):
    """Silently zipping mismatched lists would mis-pair vectors with chunks —
    a corpus-wide corruption that retrieval could not detect."""

    async def short_embed(texts):
        return [[0.0] * 3]  # one vector regardless of input count

    monkeypatch.setattr(ingestion, "embed_texts_async", short_embed)
    db = FakeSession()
    doc = "\n\n".join(f"Paragraph {i} with enough words to matter." for i in range(300))
    with pytest.raises(ValueError, match="vectors for"):
        await ingest_document(db, user_id=USER, text=doc)


# ── Happy path ──────────────────────────────────────────────

async def test_ingest_creates_document_and_chunks(patched):
    db = FakeSession()
    doc = "# Title\n\nFirst paragraph here.\n\n## Section\n\nSecond paragraph here.\n"

    result = await ingest_document(db, user_id=USER, text=doc, title="Doc")

    assert result.was_deduplicated is False
    assert result.chunk_count == len(db.chunks) > 0
    assert len(db.documents) == 1

    document = db.documents[0]
    assert document.user_id == USER
    assert document.status == "completed"
    assert document.checksum == compute_checksum(doc)
    assert document.token_count > 0


async def test_chunk_indexes_are_contiguous_from_zero(patched):
    db = FakeSession()
    doc = "\n\n".join(f"Paragraph {i} with a reasonable number of words." for i in range(300))
    await ingest_document(db, user_id=USER, text=doc)
    assert [c.chunk_index for c in db.chunks] == list(range(len(db.chunks)))


async def test_chunk_offsets_round_trip_to_the_source(patched):
    db = FakeSession()
    doc = "# H\n\nAlpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.\n"
    await ingest_document(db, user_id=USER, text=doc)
    for chunk in db.chunks:
        assert doc[chunk.char_start : chunk.char_end] == chunk.text


# ── Tenant scoping — regression guards for §5 invariants ────

async def test_every_chunk_carries_the_owning_user_id(patched):
    """chunks.user_id is the column the retrieval filter runs against. A chunk
    missing it is invisible to its owner and exposed to an unfiltered scan."""
    db = FakeSession()
    doc = "\n\n".join(f"Paragraph {i} of the document body." for i in range(300))
    await ingest_document(db, user_id=USER, text=doc)

    assert len(db.chunks) > 1
    assert all(c.user_id == USER for c in db.chunks)
    assert not any(c.user_id is None for c in db.chunks)


async def test_chunk_user_id_matches_document_user_id(patched):
    db = FakeSession()
    await ingest_document(db, user_id=OTHER_USER, text="Some body text here.")
    document = db.documents[0]
    assert all(c.user_id == document.user_id for c in db.chunks)


async def test_user_id_is_required():
    """Tenant scope is a required argument, never defaulted or inferred."""
    with pytest.raises(TypeError):
        await ingest_document(FakeSession(), text="body")  # type: ignore[call-arg]


# ── Idempotency ─────────────────────────────────────────────

async def test_reingesting_identical_content_is_deduplicated(patched):
    existing = Document(
        id=uuid.uuid4(),
        user_id=USER,
        checksum=compute_checksum("body"),
        content_type="text/plain",
        status="completed",
        token_count=7,
    )
    db = FakeSession(existing_document=existing)

    result = await ingest_document(db, user_id=USER, text="body")

    assert result.was_deduplicated is True
    assert result.document_id == existing.id
    assert db.documents == [], "dedup must not create a second document"
    assert db.chunks == [], "dedup must not re-chunk or re-embed"
    assert patched["embed"] == [], "dedup must not spend on embeddings"


def test_checksum_is_content_addressed():
    assert compute_checksum("a") == compute_checksum("a")
    assert compute_checksum("a") != compute_checksum("b")
    assert len(compute_checksum("a")) == 64


# ── Batching ────────────────────────────────────────────────

async def test_embeddings_are_requested_in_one_batched_call(patched):
    """One call for the whole document, not one per chunk — the plan is
    explicit that per-chunk HTTP round-trips dominate ingest latency."""
    db = FakeSession()
    doc = "\n\n".join(f"Paragraph {i} with several words in it." for i in range(300))

    await ingest_document(db, user_id=USER, text=doc)

    assert len(patched["embed"]) == 1, f"expected 1 batched call, got {len(patched['embed'])}"
    assert len(patched["embed"][0]) == len(db.chunks)


async def test_embedding_input_includes_the_heading_path(patched):
    """What gets embedded is heading + body, so an isolated chunk carries the
    context its position gave it."""
    db = FakeSession()
    await ingest_document(db, user_id=USER, text="# Topic\n\nThe body text.\n", title="Doc")

    embedded = patched["embed"][0]
    assert any(text.startswith("Doc > Topic") for text in embedded)
    # The stored text stays clean — the prefix is an embedding-time concern.
    assert all(not c.text.startswith("Doc > Topic") for c in db.chunks)
