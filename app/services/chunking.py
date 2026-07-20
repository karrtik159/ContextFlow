"""
Structure-first, token-aware document chunking.

Pure functions only — no I/O, no model loading, no config reads. The token
counter is injected so this module can be exhaustively unit-tested without a
tokenizer, and so the caller stays responsible for using the counter that
matches the configured embedding model.

The algorithm, in order:

1. Split on document structure — headings first, then fenced code blocks and
   paragraphs, then sentences. Never a naive fixed-width character split.
2. Pack units into chunks up to a token budget, never splitting mid-sentence
   unless a single unit is itself over budget.
3. Overlap adjacent chunks by a token budget so a fact spanning a boundary
   survives in at least one chunk.
4. Prepend the heading path to the embedded text so an isolated chunk carries
   the context its position in the document gave it.

Two invariants this module enforces (docs/ADVANCED_RAG_PLAN.md §5.6):

- NO SILENT TRUNCATION. A unit larger than the budget is hard-split and flagged
  via `ChunkDraft.was_hard_split`, never quietly cut.
- The token budget applies to what is actually EMBEDDED — body plus heading
  prefix — not to the body alone. Budgeting the body would let the prefix push
  the real input past the encoder's ceiling, where MiniLM truncates silently
  and OpenAI errors.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

# ATX markdown headings: "## Section". Leading whitespace is allowed; the hash
# run must be followed by a space so "#hashtag" is not a heading.
_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

# Fence open/close for code blocks — ``` or ~~~.
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")

# Sentence boundaries. The first alternative consumes trailing whitespace after
# western terminal punctuation; the second is a zero-width boundary after CJK
# terminal punctuation, which is not followed by a space.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")

_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")

_HEADING_SEPARATOR = " > "
_PREFIX_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ChunkDraft:
    """One chunk, before embedding or persistence.

    `char_start`/`char_end` index the ORIGINAL document text, so
    `original[char_start:char_end] == text` always holds. Overlap means
    consecutive drafts' spans intentionally intersect.
    """

    text: str
    heading_path: str | None
    char_start: int
    char_end: int
    token_count: int
    was_hard_split: bool = False

    @property
    def embedding_input(self) -> str:
        """Exactly what gets embedded — heading context plus body.

        `token_count` measures THIS string, not `text`.
        """
        if self.heading_path:
            return f"{self.heading_path}{_PREFIX_SEPARATOR}{self.text}"
        return self.text


@dataclass(frozen=True)
class _Unit:
    """A span of the source document that packing treats as indivisible."""

    start: int
    end: int
    heading_path: str | None
    tokens: int
    # True when this unit came from _hard_split — i.e. we cut inside a sentence
    # because it had no remaining internal structure to exploit.
    hard: bool = False


def _prefix_tokens(heading_path: str | None, count_tokens: Callable[[str], int]) -> int:
    """Tokens consumed by the heading prefix that `embedding_input` prepends."""
    if not heading_path:
        return 0
    return count_tokens(f"{heading_path}{_PREFIX_SEPARATOR}")


def _iter_sections(text: str, title: str | None) -> Iterator[tuple[int, int, str | None]]:
    """Yield (body_start, body_end, heading_path) covering the document.

    Maintains a heading stack so nested sections produce
    "Title > Section > Subsection". Headings inside fenced code blocks are
    ignored — a '#' in a Python snippet is a comment, not a section.
    """
    stack: list[tuple[int, str]] = []
    body_start = 0
    in_fence = False
    fence_marker = ""
    pos = 0

    def current_path() -> str | None:
        parts = ([title] if title else []) + [t for _, t in stack]
        return _HEADING_SEPARATOR.join(parts) if parts else None

    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        stripped = line.rstrip("\n")

        fence = _FENCE_RE.match(stripped)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            continue

        if in_fence:
            continue

        heading = _HEADING_RE.match(stripped)
        if not heading:
            continue

        if line_start > body_start:
            yield body_start, line_start, current_path()

        level = len(heading.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading.group(2).strip()))
        body_start = pos

    if body_start < len(text):
        yield body_start, len(text), current_path()


def _iter_blocks(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) blocks within a section.

    A fenced code block is one block regardless of blank lines inside it —
    splitting a code sample on a blank line produces two useless fragments.
    Otherwise blocks are blank-line-separated paragraphs.
    """
    body = text[start:end]
    fence_spans: list[tuple[int, int]] = []
    in_fence = False
    fence_start = 0
    fence_marker = ""
    pos = 0

    for line in body.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        fence = _FENCE_RE.match(line.rstrip("\n"))
        if not fence:
            continue
        marker = fence.group(1)[0]
        if not in_fence:
            in_fence, fence_start, fence_marker = True, line_start, marker
        elif marker == fence_marker:
            fence_spans.append((fence_start, pos))
            in_fence = False
    if in_fence:  # unterminated fence — treat the remainder as one block
        fence_spans.append((fence_start, len(body)))

    cursor = 0
    for fence_start, fence_end in fence_spans:
        if fence_start > cursor:
            yield from _iter_paragraphs(body, cursor, fence_start, start)
        if body[fence_start:fence_end].strip():
            yield start + fence_start, start + fence_end
        cursor = fence_end
    if cursor < len(body):
        yield from _iter_paragraphs(body, cursor, len(body), start)


def _iter_paragraphs(body: str, start: int, end: int, base: int) -> Iterator[tuple[int, int]]:
    segment = body[start:end]
    cursor = 0
    for match in _BLANK_LINE_RE.finditer(segment):
        if segment[cursor : match.start()].strip():
            yield base + start + cursor, base + start + match.start()
        cursor = match.end()
    if segment[cursor:].strip():
        yield base + start + cursor, base + start + len(segment)


def _iter_sentences(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    segment = text[start:end]
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(segment):
        if segment[cursor : match.start()].strip():
            yield start + cursor, start + match.start()
        cursor = match.end()
    if segment[cursor:].strip():
        yield start + cursor, start + len(segment)


def _prefix_for(heading_path: str | None) -> str:
    return f"{heading_path}{_PREFIX_SEPARATOR}" if heading_path else ""


def _hard_split(
    text: str,
    start: int,
    end: int,
    budget: int,
    prefix: str,
    count_tokens: Callable[[str], int],
) -> Iterator[tuple[int, int]]:
    """Split an over-budget unit that has no internal structure left to exploit.

    Reached by a single sentence longer than the budget, or by CJK / minified
    text with no whitespace at all. Estimates a character budget from the
    measured token density, then shrinks until it verifiably fits — so it never
    relies on a hardcoded chars-per-token ratio, which varies by script and by
    tokenizer.
    """
    cursor = start
    while cursor < end:
        remaining = text[cursor:end]
        tokens = count_tokens(prefix + remaining)
        if tokens <= budget:
            yield cursor, end
            return

        chars = max(1, int(len(remaining) * budget / max(1, tokens)))
        while chars > 1 and count_tokens(prefix + text[cursor : cursor + chars]) > budget:
            chars = max(1, int(chars * 0.8))
        yield cursor, cursor + chars
        cursor += chars


def _build_units(
    text: str,
    title: str | None,
    target_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[_Unit]:
    """Decompose the document into packable units, descending through structure
    only as far as each block's size requires.

    Budgets are measured against `prefix + span`, never against the span alone:
    the prefix is part of what gets embedded, and token counts are not additive,
    so `count(prefix) + count(span)` is not `count(prefix + span)`.
    """
    units: list[_Unit] = []
    for sec_start, sec_end, heading_path in _iter_sections(text, title):
        prefix = _prefix_for(heading_path)
        if heading_path and _prefix_tokens(heading_path, count_tokens) >= target_tokens:
            raise ValueError(
                f"Heading path {heading_path!r} alone consumes the entire "
                f"{target_tokens}-token chunk budget, leaving no room for content. "
                f"Raise CHUNK_TARGET_TOKENS or shorten the document title/headings."
            )

        def fits(span_start: int, span_end: int) -> bool:
            return count_tokens(prefix + text[span_start:span_end]) <= target_tokens

        for block_start, block_end in _iter_blocks(text, sec_start, sec_end):
            if fits(block_start, block_end):
                units.append(
                    _Unit(block_start, block_end, heading_path, count_tokens(text[block_start:block_end]))
                )
                continue

            for sent_start, sent_end in _iter_sentences(text, block_start, block_end):
                if fits(sent_start, sent_end):
                    units.append(
                        _Unit(sent_start, sent_end, heading_path, count_tokens(text[sent_start:sent_end]))
                    )
                    continue
                for piece_start, piece_end in _hard_split(
                    text, sent_start, sent_end, target_tokens, prefix, count_tokens
                ):
                    units.append(
                        _Unit(
                            piece_start,
                            piece_end,
                            heading_path,
                            count_tokens(text[piece_start:piece_end]),
                            hard=True,
                        )
                    )
    return units


def chunk_document(
    text: str,
    *,
    count_tokens: Callable[[str], int],
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    title: str | None = None,
) -> list[ChunkDraft]:
    """Split `text` into overlapping, structure-aware, token-bounded chunks.

    Args:
        count_tokens: Tokenizer for the CONFIGURED embedding model. Using a
            different tokenizer here than at embedding time reintroduces exactly
            the silent-truncation bug the budget exists to prevent.
        target_tokens: Soft budget per chunk, measured over `embedding_input`.
        overlap_tokens: Tokens of trailing context repeated into the next chunk.
        max_tokens: Hard ceiling of the embedding model. Chunks are built
            against `target_tokens`; this is the assertion boundary.
        title: Document title, seeded as the root of every heading path.

    Returns an empty list for blank input.

    Raises:
        ValueError: on a nonsensical budget, or if a produced chunk still
            exceeds `max_tokens` — a bug here must fail loudly rather than hand
            the encoder input it would silently truncate.
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be non-negative, got {overlap_tokens}")
    if overlap_tokens >= target_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be less than target_tokens "
            f"({target_tokens}) — equal or greater cannot make forward progress."
        )
    if target_tokens > max_tokens:
        raise ValueError(
            f"target_tokens ({target_tokens}) exceeds the embedding model's "
            f"max_tokens ({max_tokens})."
        )

    if not text.strip():
        return []

    units = _build_units(text, title, target_tokens, count_tokens)
    if not units:
        return []

    def measure(group: list[_Unit]) -> int:
        """Tokens of the chunk this group would produce.

        Measured on the assembled string rather than summed per unit. Token
        counts are NOT additive — subword tokenizers merge across boundaries,
        and for scripts without whitespace the difference is large enough to
        blow past the model ceiling. Summing was the original bug here.
        """
        start, end = group[0].start, group[-1].end
        return count_tokens(_prefix_for(group[0].heading_path) + text[start:end])

    def emit(group: list[_Unit]) -> ChunkDraft:
        start, end = group[0].start, group[-1].end
        body = text[start:end]
        # A chunk carries the heading path of its FIRST unit. Overlap can pull
        # in the tail of a previous section; attributing the chunk to where it
        # starts keeps the path stable and truthful for citations.
        heading = group[0].heading_path
        token_count = measure(group)
        if token_count > max_tokens:
            raise ValueError(
                f"Chunk at chars [{start}:{end}] is {token_count} tokens, over the "
                f"model ceiling of {max_tokens}. This is a chunker bug — refusing to "
                f"emit a chunk the encoder would silently truncate."
            )
        return ChunkDraft(
            text=body,
            heading_path=heading,
            char_start=start,
            char_end=end,
            token_count=token_count,
            was_hard_split=any(u.hard for u in group),
        )

    def carry_from(group: list[_Unit]) -> list[_Unit]:
        """Trailing units worth roughly `overlap_tokens`, for the next chunk."""
        if overlap_tokens <= 0:
            return []
        carry: list[_Unit] = []
        carried = 0
        for unit in reversed(group):
            if carried >= overlap_tokens:
                break
            carry.insert(0, unit)
            carried += unit.tokens
        # Never carry the whole group forward: the next chunk would re-emit the
        # same span and packing would not make progress.
        if len(carry) >= len(group):
            carry = carry[1:]
        return carry

    drafts: list[ChunkDraft] = []
    pending: list[_Unit] = []

    for unit in units:
        # A unit from a different section carries a different heading path, and
        # a chunk can only claim one. Break rather than mislabel the tail.
        section_changed = bool(pending) and unit.heading_path != pending[-1].heading_path
        if pending and (section_changed or measure(pending + [unit]) > target_tokens):
            drafts.append(emit(pending))
            pending = [] if section_changed else carry_from(pending)
            # Carried units belong to the previous section's heading; if the
            # carry alone plus the new unit already overflows, drop the carry.
            if pending and measure(pending + [unit]) > target_tokens:
                pending = []
        pending.append(unit)

    # `pending` always holds at least the unit appended on the final iteration,
    # so this is never a duplicate of the chunk emitted just above.
    if pending:
        drafts.append(emit(pending))

    return drafts
