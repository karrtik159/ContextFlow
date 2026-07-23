"""
Token-budget helpers (Phase 9).

`head_within_token_budget` bounds a single embedding input; `build_context_block`
bounds the assembled prompt. Both replace an implicit "trust the size" with an
explicit, measured, logged limit — the same discipline the chunker applies to
the corpus, extended to the non-chunked paths.

The tokenizer here is a word counter injected via monkeypatch, so these tests
need no model.
"""

from __future__ import annotations

import uuid

import pytest

from app.services import embeddings
from app.services.retrieval import pipeline as pipeline_mod
from app.services.retrieval.contracts import RetrievedChunk


def _word_count(text: str) -> int:
    return len(text.split())


# ── head_within_token_budget ────────────────────────────────


def test_under_budget_text_is_returned_whole(monkeypatch):
    monkeypatch.setattr(embeddings, "count_tokens", _word_count)
    head, truncated = embeddings.head_within_token_budget("one two three", max_tokens=10)
    assert head == "one two three"
    assert truncated is False


def test_over_budget_text_is_cut_and_flagged(monkeypatch):
    monkeypatch.setattr(embeddings, "count_tokens", _word_count)
    text = " ".join(f"w{i}" for i in range(100))
    head, truncated = embeddings.head_within_token_budget(text, max_tokens=10)
    assert truncated is True
    assert _word_count(head) <= 10
    assert text.startswith(head), "the head must be a genuine prefix"


def test_empty_text_is_never_truncated(monkeypatch):
    monkeypatch.setattr(embeddings, "count_tokens", _word_count)
    assert embeddings.head_within_token_budget("", max_tokens=5) == ("", False)


def test_nonpositive_budget_is_rejected(monkeypatch):
    monkeypatch.setattr(embeddings, "count_tokens", _word_count)
    with pytest.raises(ValueError, match="max_tokens"):
        embeddings.head_within_token_budget("x", max_tokens=0)


# ── build_context_block token budget ────────────────────────


def _chunk(label: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source="vector", rank=int(label.strip("[]")), score=0.9,
        chunk_id=uuid.uuid4(), citation_label=label,
    )


def test_empty_chunks_render_the_no_context_sentinel():
    assert pipeline_mod.build_context_block([]) == "NO CONTEXT RETRIEVED."


def test_all_blocks_included_when_under_budget(monkeypatch):
    monkeypatch.setattr("app.services.embeddings.count_tokens", _word_count)
    chunks = [_chunk("[1]", "alpha beta"), _chunk("[2]", "gamma delta")]
    block = pipeline_mod.build_context_block(chunks, token_budget=1000)
    assert "<<<CONTEXT [1]" in block
    assert "<<<CONTEXT [2]" in block


def test_budget_trims_lower_ranked_blocks(monkeypatch):
    monkeypatch.setattr("app.services.embeddings.count_tokens", _word_count)
    chunks = [
        _chunk("[1]", "aaa bbb ccc"),
        _chunk("[2]", "ddd eee fff"),
        _chunk("[3]", "ggg hhh iii"),
    ]
    # A budget that admits the first block's rendered size but not all three.
    first_cost = _word_count(pipeline_mod._render_block(chunks[0]))
    block = pipeline_mod.build_context_block(chunks, token_budget=first_cost + 1)
    assert "<<<CONTEXT [1]" in block
    assert "<<<CONTEXT [3]" not in block


def test_first_block_is_kept_even_if_it_alone_exceeds_budget(monkeypatch):
    """Returning no context when retrieval found something is a worse failure
    than a long prompt."""
    monkeypatch.setattr("app.services.embeddings.count_tokens", _word_count)
    chunks = [_chunk("[1]", "aaa bbb ccc ddd eee fff ggg")]
    block = pipeline_mod.build_context_block(chunks, token_budget=1)
    assert "<<<CONTEXT [1]" in block
