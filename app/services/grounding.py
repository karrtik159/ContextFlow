"""
Post-hoc citation validation — the Phase 4 exit criterion.

The synthesis prompt instructs the model to cite every claim with the bracket
label of the context block it came from. Until now nothing checked the output:
an answer citing "[7]" when five blocks were supplied rendered a dangling
reference to the user, and — worse — a confabulated citation is positive
evidence the model was not reading its context, yet the answer was still
marked grounded and cached.

Policy, applied in `rag_service` after synthesis:

- A cited label that RESOLVES to a supplied block: fine.
- A cited label that resolves to NOTHING: stripped from the answer (repair),
  and the answer is marked ungrounded — reject-for-caching, not
  reject-for-the-user. The prose usually survives its bad footnote; freezing
  it into the semantic cache must not.
- ZERO citations: ungrounded. Either the model ignored the prompt or it is
  hedging ("the context does not contain..."); both are fine to return and
  wrong to cache.

`grounded` therefore now means: the answer cites, and every citation resolves.
`should_cache` already requires it (§5.9) — this module is what makes the flag
earned rather than assumed.

Known tradeoff: the label pattern is `[<digits>]`, so a bracketed index in a
code-style answer ("arr[0]") that matches no supplied label is stripped.
Supplied labels are 1..k (k = RETRIEVAL_TOP_K), answers are conversational and
capped at ~150 words by the prompt, so this is accepted rather than parsed
around. Revisit if answers ever legitimately carry bracketed literals.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.retrieval.contracts import RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")

# Repair leftovers: doubled spaces where a label was lifted out, and a space
# stranded before punctuation ("fact  ." / "fact ."  → "fact.").
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([.,;:!?])")


@dataclass(frozen=True)
class CitationCheck:
    """The verdict on one synthesized answer."""

    grounded: bool
    answer: str  # repaired — identical to the input when nothing was stripped
    cited: tuple[str, ...]
    resolvable: tuple[str, ...]
    unresolvable: tuple[str, ...]

    @property
    def was_repaired(self) -> bool:
        return bool(self.unresolvable)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "cited": list(self.cited),
            "unresolvable": list(self.unresolvable),
            "repaired": self.was_repaired,
        }


def validate_citations(answer: str, chunks: Sequence[RetrievedChunk]) -> CitationCheck:
    """Check every `[n]` in `answer` against the supplied context blocks.

    `chunks` are the blocks the synthesis model was shown — the labels were
    stamped by `assign_citation_labels`, so this resolves against exactly what
    the model could legitimately cite.
    """
    supplied = {chunk.citation_label for chunk in chunks if chunk.citation_label}

    cited: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        label = f"[{match.group(1)}]"
        if label not in cited:
            cited.append(label)

    resolvable = tuple(label for label in cited if label in supplied)
    unresolvable = tuple(label for label in cited if label not in supplied)

    repaired = answer
    if unresolvable:
        for label in unresolvable:
            repaired = repaired.replace(label, "")
        repaired = _MULTISPACE_RE.sub(" ", repaired)
        repaired = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", repaired).strip()

    return CitationCheck(
        grounded=bool(resolvable) and not unresolvable,
        answer=repaired,
        cited=tuple(cited),
        resolvable=resolvable,
        unresolvable=unresolvable,
    )
