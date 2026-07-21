"""
Context sufficiency — deciding whether to answer at all.

Phase 3 measured, twice, that this cannot be done with a relevance score.
Neither a cosine floor (`RETRIEVAL_MIN_SIMILARITY`) nor a cross-encoder floor
(`RERANK_MIN_SCORE`) separates answerable from unanswerable queries: abstention
never exceeded 1/5 at any threshold while recall collapsed to 0.860, and the
unanswerable "Which OIDC claims does Meridian map to workspace roles?" scored
0.992 on the reranker — above most correct answers.

That is not a tuning failure, it is a category error. Google / UC San Diego,
*Sufficient Context: A New Lens on RAG Systems* (ICLR 2025, arXiv 2411.06037),
names the distinction: context is SUFFICIENT if it contains everything needed
for a definitive answer, and sufficiency is not relevance. A cross-encoder
scores aboutness, and an unanswerable question about a documented subject is
maximally about it. The same paper reports the failure mode seen here — RAG
*reduces* a model's willingness to abstain.

The categories of unanswerable request come from UAEval4RAG (arXiv 2412.12300):
underspecified, false-presupposition, nonsensical, modality-limited,
safety-concerned, and out-of-database. They are listed because they are NOT one
problem: a mechanism can be perfect on one and useless on another, and only a
per-category measurement shows which.

This module deliberately starts with the cheapest detector that could work, so
that the expensive ones are added against evidence rather than assumption.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.services.retrieval.contracts import RetrievedChunk

logger = logging.getLogger(__name__)

# Function words and non-domain verbs. Coverage keys on rare, content-bearing
# tokens: a query is not unanswerable because it contains "happens".
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from
    get got had has have how i if in into is it its may might must not of off
    on one only or our out per should show so some such than that the their
    then there these they this those to two use used using via was were what
    when where which who why will with would you your make makes need needs
    want happens happen occurs switch stop instead rather result results just
    also same still about does doesn't don't me my we us
    """.split()
)

# Starts on any alphanumeric so that identifiers beginning with a digit —
# release tags like `20260714-af`, error codes, versions — are captured. An
# earlier `[A-Za-z]` anchor silently skipped exactly the tokens this check
# exists to notice.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PURE_NUMBER = re.compile(r"^\d+$")


@dataclass
class SufficiencyVerdict:
    """Whether the retrieved context can support an answer.

    `sufficient=True` is NOT a claim that the answer is correct — only that
    abstaining would be wrong. The honest-empty path is the other branch.
    """

    sufficient: bool
    method: str
    reason: str
    uncovered_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "method": self.method,
            "reason": self.reason,
            "uncovered_terms": self.uncovered_terms,
            **self.metadata,
        }


def salient_terms(query: str) -> list[str]:
    """Content-bearing tokens of a query, lowercased and de-duplicated."""
    seen: set[str] = set()
    terms: list[str] = []
    for match in _TOKEN.finditer(query):
        token = match.group(0).casefold()
        if len(token) < 3 or token in _STOPWORDS:
            continue
        # A bare number carries no topic. "thirty days" is about days.
        if _PURE_NUMBER.match(token):
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


async def assess_sufficiency(
    db,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    user_id,
) -> SufficiencyVerdict:
    """The sufficiency gate as configured.

    Coverage is checked against the whole tenant CORPUS, not against the
    retrieved chunks. Checking the chunks produces constant false refusals —
    an ordinary word like "often" is missing from any given five chunks — and
    wrongly refusing a real question is the worse error by a wide margin.
    Corpus-wide is also the honest reading of "we have nothing on this".

    Returns `sufficient=True` when the gate is disabled, so callers never
    branch on the flag themselves and the trace always carries a verdict.
    Any failure degrades to sufficient: this gate exists to suppress bad
    answers, and it must never become a new way to lose good ones.
    """
    if not settings.SUFFICIENCY_ENABLED:
        return SufficiencyVerdict(
            sufficient=True, method="disabled", reason="SUFFICIENCY_ENABLED is false"
        )

    if not chunks:
        return SufficiencyVerdict(
            sufficient=False, method="term_coverage", reason="no context retrieved"
        )

    terms = salient_terms(query)
    if not terms:
        return SufficiencyVerdict(
            sufficient=True, method="term_coverage", reason="no salient terms to check"
        )

    try:
        from app.services.retrieval.sources import find_uncovered_terms

        uncovered = await find_uncovered_terms(
            db, terms=terms, user_id=user_id, ts_config=settings.SPARSE_TS_CONFIG
        )
    except Exception as exc:
        logger.warning("Sufficiency check failed, not abstaining: %s", exc, exc_info=True)
        return SufficiencyVerdict(
            sufficient=True,
            method="term_coverage",
            reason=f"check failed, defaulted to sufficient: {type(exc).__name__}",
        )

    max_uncovered = settings.SUFFICIENCY_MAX_UNCOVERED_TERMS
    sufficient = len(uncovered) <= max_uncovered
    return SufficiencyVerdict(
        sufficient=sufficient,
        method="term_coverage",
        reason=(
            "every salient query term occurs somewhere in the corpus"
            if sufficient
            else f"{len(uncovered)} query term(s) absent from the corpus"
        ),
        uncovered_terms=uncovered,
        metadata={"salient_terms": terms, "max_uncovered": max_uncovered},
    )
