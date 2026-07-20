"""
Unit tests for the structure-first chunker.

Fully deterministic and model-free: `count_tokens` is injected, so these tests
never load a tokenizer. `_word_tokens` is a stand-in whose only requirement is
monotonicity — the chunker must not depend on any particular tokenizer.

The universal invariants (offsets round-trip, budgets hold, overlap makes
forward progress) are asserted by `assert_chunk_invariants`, which every
structural test calls. That is deliberate: it is the property that must hold for
inputs nobody thought to write a test for.
"""

from __future__ import annotations

import pytest

from app.services.chunking import ChunkDraft, chunk_document


def _word_tokens(text: str) -> int:
    """Whitespace word count, with a per-character floor for scripts that do not
    use spaces — otherwise a 5,000-character CJK paragraph counts as 1 token and
    every budget check trivially passes."""
    words = len(text.split())
    return max(words, len(text) // 4)


def _chunk(text: str, *, target=50, overlap=10, max_tokens=200, title=None) -> list[ChunkDraft]:
    return chunk_document(
        text,
        count_tokens=_word_tokens,
        target_tokens=target,
        overlap_tokens=overlap,
        max_tokens=max_tokens,
        title=title,
    )


def assert_chunk_invariants(drafts: list[ChunkDraft], source: str, max_tokens: int) -> None:
    """Properties that must hold for ANY input, not just the ones tested here."""
    for i, d in enumerate(drafts):
        assert source[d.char_start : d.char_end] == d.text, (
            f"chunk {i} offsets do not round-trip to the source"
        )
        assert d.char_start < d.char_end, f"chunk {i} has an empty or inverted span"
        assert d.text.strip(), f"chunk {i} is whitespace-only"
        assert d.token_count <= max_tokens, (
            f"chunk {i} is {d.token_count} tokens, over the {max_tokens} ceiling"
        )
        assert d.token_count == _word_tokens(d.embedding_input), (
            f"chunk {i} token_count must measure embedding_input, not text"
        )

    # Chunks advance through the document and never repeat a span exactly.
    for a, b in zip(drafts, drafts[1:]):
        assert b.char_start > a.char_start, "chunks must make forward progress"
        assert b.char_end > a.char_end, "chunk ends must advance"


# ── Empty and degenerate input ──────────────────────────────

@pytest.mark.parametrize("blank", ["", "   ", "\n\n\n", "\t \n  \t"])
def test_blank_input_yields_no_chunks(blank: str):
    assert _chunk(blank) == []


def test_single_word():
    drafts = _chunk("hello")
    assert len(drafts) == 1
    assert drafts[0].text == "hello"
    assert_chunk_invariants(drafts, "hello", 200)


# ── Budget validation ───────────────────────────────────────

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target": 0}, "target_tokens must be positive"),
        ({"target": -5}, "target_tokens must be positive"),
        ({"overlap": -1}, "overlap_tokens must be non-negative"),
        ({"target": 10, "overlap": 10}, "must be less than target_tokens"),
        ({"target": 10, "overlap": 20}, "must be less than target_tokens"),
        ({"target": 500, "max_tokens": 100}, "exceeds the embedding model"),
    ],
)
def test_invalid_budgets_raise(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        _chunk("some text here", **kwargs)


# ── Structure: headings ─────────────────────────────────────

def test_heading_path_is_built_and_nested():
    doc = (
        "# Guide\n\nIntro paragraph.\n\n"
        "## Setup\n\nInstall the thing.\n\n"
        "### Windows\n\nRun the installer.\n\n"
        "## Usage\n\nCall the function.\n"
    )
    drafts = _chunk(doc, target=20, overlap=0)
    assert_chunk_invariants(drafts, doc, 200)

    paths = [d.heading_path for d in drafts]
    assert "Guide" in paths
    assert "Guide > Setup" in paths
    assert "Guide > Setup > Windows" in paths
    # A same-level heading pops the deeper one rather than nesting under it.
    assert "Guide > Usage" in paths
    assert not any("Windows > Usage" in (p or "") for p in paths)


def test_title_seeds_the_heading_path():
    doc = "# Section\n\nBody text here.\n"
    drafts = _chunk(doc, title="My Doc")
    assert drafts[0].heading_path == "My Doc > Section"


def test_title_used_alone_when_no_headings():
    doc = "Just a plain paragraph with no headings at all.\n"
    drafts = _chunk(doc, title="My Doc")
    assert drafts[0].heading_path == "My Doc"


def test_heading_path_is_prepended_to_embedding_input_only():
    doc = "# Topic\n\nThe body.\n"
    d = _chunk(doc, title="Doc")[0]
    assert d.heading_path not in d.text
    assert d.embedding_input.startswith("Doc > Topic")
    assert d.text.strip() == "The body."


def test_hash_without_space_is_not_a_heading():
    doc = "#hashtag is not a heading and neither is #1 in a list.\n"
    drafts = _chunk(doc)
    assert drafts[0].heading_path is None


# ── Structure: code blocks ──────────────────────────────────

def test_fenced_code_block_is_not_split_on_blank_lines():
    doc = (
        "# Example\n\n"
        "Here is code:\n\n"
        "```python\n"
        "def a():\n"
        "    pass\n"
        "\n"
        "def b():\n"
        "    pass\n"
        "```\n\n"
        "After the code.\n"
    )
    drafts = _chunk(doc, target=200, overlap=0, max_tokens=500)
    assert_chunk_invariants(drafts, doc, 500)
    # The whole fence lands in one chunk, blank line included.
    holder = [d for d in drafts if "def a()" in d.text]
    assert len(holder) == 1
    assert "def b()" in holder[0].text


def test_heading_inside_code_fence_is_ignored():
    doc = (
        "# Real Heading\n\n"
        "```python\n"
        "# This is a comment, not a heading\n"
        "x = 1\n"
        "```\n\n"
        "Text after.\n"
    )
    drafts = _chunk(doc, target=200, max_tokens=500)
    for d in drafts:
        assert "comment" not in (d.heading_path or "")
        assert d.heading_path == "Real Heading"


def test_unterminated_fence_does_not_lose_content():
    doc = "# H\n\n```python\nx = 1\ny = 2\n"
    drafts = _chunk(doc, target=200, max_tokens=500)
    assert_chunk_invariants(drafts, doc, 500)
    assert any("y = 2" in d.text for d in drafts)


def test_tilde_fence_is_supported():
    doc = "~~~\ncode here\n\nmore code\n~~~\n"
    drafts = _chunk(doc, target=200, max_tokens=500)
    assert len(drafts) == 1
    assert "more code" in drafts[0].text


# ── Structure: tables ───────────────────────────────────────

def test_markdown_table_rows_stay_together():
    doc = (
        "# Data\n\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n"
    )
    drafts = _chunk(doc, target=200, max_tokens=500)
    assert len(drafts) == 1, "a table with no blank lines is one paragraph block"
    assert "| 3 | 4 |" in drafts[0].text


# ── Token budgets and packing ───────────────────────────────

def test_chunks_respect_the_target_budget():
    doc = "\n\n".join(f"Paragraph number {i} with several words in it." for i in range(40))
    drafts = _chunk(doc, target=30, overlap=0)
    assert len(drafts) > 1
    assert_chunk_invariants(drafts, doc, 200)
    for d in drafts:
        assert d.token_count <= 30 + 10, "chunk substantially over the target budget"


def test_prefix_counts_against_the_budget():
    """A long heading path must shrink the body budget, not be added on top."""
    long_title = " ".join(f"word{i}" for i in range(20))
    doc = "# Section\n\n" + "\n\n".join(f"Body paragraph {i} here." for i in range(20))
    drafts = _chunk(doc, target=40, overlap=0, max_tokens=45, title=long_title)
    # max_tokens is only barely above target: if the prefix were added on top of
    # a full body, emit() would raise. Reaching here means it was budgeted.
    assert_chunk_invariants(drafts, doc, 45)


def test_sentence_is_never_split_when_it_fits():
    doc = " ".join(f"Sentence number {i} ends here." for i in range(30))
    drafts = _chunk(doc, target=25, overlap=0)
    assert_chunk_invariants(drafts, doc, 200)
    for d in drafts:
        assert not d.was_hard_split
        assert d.text.strip().endswith("."), "chunk ended mid-sentence"


# ── Overlap ─────────────────────────────────────────────────

def test_overlap_repeats_trailing_context():
    doc = "\n\n".join(f"Paragraph {i} with a handful of words." for i in range(20))
    with_overlap = _chunk(doc, target=30, overlap=12)
    assert_chunk_invariants(with_overlap, doc, 200)
    assert len(with_overlap) > 1
    # Consecutive chunks share source characters.
    for a, b in zip(with_overlap, with_overlap[1:]):
        assert b.char_start < a.char_end, "expected overlapping spans"


def test_zero_overlap_produces_disjoint_chunks():
    doc = "\n\n".join(f"Paragraph {i} with a handful of words." for i in range(20))
    drafts = _chunk(doc, target=30, overlap=0)
    assert_chunk_invariants(drafts, doc, 200)
    for a, b in zip(drafts, drafts[1:]):
        assert b.char_start >= a.char_end, "zero overlap must not repeat content"


def test_overlap_still_terminates_on_uniform_input():
    """Overlap that could consume a whole chunk must not stall progress."""
    doc = "\n\n".join(f"Para {i}." for i in range(30))
    drafts = _chunk(doc, target=12, overlap=11)
    assert_chunk_invariants(drafts, doc, 200)
    assert len(drafts) < 200, "packing failed to converge"


# ── Hard splits ─────────────────────────────────────────────

def test_single_oversized_paragraph_is_hard_split_and_flagged():
    doc = " ".join(f"word{i}" for i in range(5000))  # one sentence, no periods
    drafts = _chunk(doc, target=100, overlap=0, max_tokens=120)
    assert len(drafts) > 1
    assert_chunk_invariants(drafts, doc, 120)
    assert all(d.was_hard_split for d in drafts)


def test_hard_split_loses_no_characters():
    doc = " ".join(f"word{i}" for i in range(2000))
    drafts = _chunk(doc, target=100, overlap=0, max_tokens=120)
    # With zero overlap the spans tile the covered region contiguously.
    for a, b in zip(drafts, drafts[1:]):
        assert b.char_start == a.char_end, "hard split dropped characters"


def test_fifty_thousand_token_paragraph():
    """The plan's explicit edge case: one enormous structureless paragraph."""
    doc = " ".join(f"token{i}" for i in range(50_000))
    drafts = _chunk(doc, target=500, overlap=0, max_tokens=600)
    assert_chunk_invariants(drafts, doc, 600)
    assert len(drafts) >= 90


# ── CJK ─────────────────────────────────────────────────────

def test_cjk_sentences_split_on_ideographic_punctuation():
    doc = "这是第一句话。这是第二句话。这是第三句话。" * 20
    drafts = _chunk(doc, target=60, overlap=0, max_tokens=100)
    assert len(drafts) > 1
    assert_chunk_invariants(drafts, doc, 100)


def test_cjk_without_punctuation_is_hard_split():
    doc = "字" * 4000
    drafts = _chunk(doc, target=100, overlap=0, max_tokens=120)
    assert len(drafts) > 1
    assert_chunk_invariants(drafts, doc, 120)
    assert all(d.was_hard_split for d in drafts)


# ── Offset fidelity across a realistic document ─────────────

def test_offsets_round_trip_on_a_mixed_document():
    doc = (
        "# Title\n\n"
        "First paragraph with some words.\n\n"
        "## Code\n\n"
        "```js\nconst x = 1;\n\nconst y = 2;\n```\n\n"
        "## Table\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "## Prose\n\n"
        "Sentence one. Sentence two. Sentence three.\n\n"
        "日本語の文章です。これも文章です。\n"
    )
    drafts = _chunk(doc, target=40, overlap=8, max_tokens=200)
    assert_chunk_invariants(drafts, doc, 200)
    assert len(drafts) > 1


def test_crlf_line_endings_do_not_corrupt_offsets():
    doc = "# Heading\r\n\r\nParagraph one here.\r\n\r\nParagraph two here.\r\n"
    drafts = _chunk(doc, target=200, max_tokens=500)
    assert_chunk_invariants(drafts, doc, 500)
