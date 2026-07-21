"""
Golden-set loading and relevance resolution.

Relevance is labelled as (document, marker substring) rather than by chunk id,
because chunk ids are assigned at ingest and chunk boundaries move whenever the
chunker changes. Phase 3 changes chunking; labels pinned to chunk ids would all
have to be re-authored, and in practice would just be regenerated from whatever
the retriever currently returns — which makes the golden set agree with the
system by construction and measure nothing.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GOLDEN_SET = Path("tests/test_eval/fixtures/golden_set.json")

# Marker for queries the corpus deliberately cannot answer.
NOT_IN_CORPUS = "NOT_IN_CORPUS"


@dataclass(frozen=True)
class GoldenDocument:
    key: str
    title: str
    text: str


@dataclass(frozen=True)
class RelevanceLabel:
    document: str
    contains: str


# Categories of unanswerable query, from UAEval4RAG (arXiv 2412.12300).
# Recorded per-query because abstention accuracy is not one number: a system can
# be perfect on out-of-database requests and useless on underspecified ones, and
# an aggregate hides exactly that.
UNANSWERABLE_CATEGORIES = frozenset(
    {
        "underspecified",  # essential information missing from the request
        "false_presupposition",  # built on an assumption the corpus contradicts
        "nonsensical",  # well-formed vocabulary, no coherent meaning
        "modality_limited",  # asks for a format the system cannot produce
        "safety_concerned",  # fulfilling it would plausibly cause harm
        "out_of_database",  # topical, but the answer is not in the corpus
    }
)


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    relevant: list[RelevanceLabel]
    reference: str
    category: str = ""
    # True when EVERY salient term in the query appears somewhere in the
    # corpus. These are the queries that defeat a term-coverage heuristic: it
    # can only ever detect an unanswerable query by noticing a missing word, so
    # it is blind to this whole class. Labelled explicitly so the eval reports
    # the two populations separately instead of averaging a cheap win over a
    # hard loss.
    terms_all_in_corpus: bool = False

    @property
    def is_unanswerable(self) -> bool:
        """True when the corpus deliberately does not contain the answer.

        These are excluded from recall and nDCG — a query with no relevant
        chunks cannot be scored on them, and averaging in a zero would make the
        retriever look worse the more honest the golden set is. They are scored
        separately, on whether the system correctly declines.
        """
        return not self.relevant or self.reference == NOT_IN_CORPUS


@dataclass
class GoldenSet:
    documents: list[GoldenDocument] = field(default_factory=list)
    queries: list[GoldenQuery] = field(default_factory=list)

    @property
    def answerable(self) -> list[GoldenQuery]:
        return [q for q in self.queries if not q.is_unanswerable]

    @property
    def unanswerable(self) -> list[GoldenQuery]:
        return [q for q in self.queries if q.is_unanswerable]


def load_golden_set(path: str | Path = DEFAULT_GOLDEN_SET) -> GoldenSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    documents = [
        GoldenDocument(key=d["key"], title=d["title"], text=d["text"])
        for d in raw["documents"]
    ]
    known_keys = {d.key for d in documents}

    queries: list[GoldenQuery] = []
    for q in raw["queries"]:
        labels = [
            RelevanceLabel(document=r["document"], contains=r["contains"])
            for r in q.get("relevant", [])
        ]
        for label in labels:
            if label.document not in known_keys:
                raise ValueError(
                    f"Query {q['id']!r} labels document {label.document!r}, "
                    f"which is not in the golden set."
                )
        category = q.get("category", "")
        if category and category not in UNANSWERABLE_CATEGORIES:
            raise ValueError(
                f"Query {q['id']!r} has category {category!r}, which is not one of "
                f"{sorted(UNANSWERABLE_CATEGORIES)}. A typo here would silently "
                f"create a category that every per-category metric reports as empty."
            )
        queries.append(
            GoldenQuery(
                id=q["id"],
                query=q["query"],
                relevant=labels,
                reference=q.get("reference", ""),
                category=category,
                terms_all_in_corpus=q.get("terms_all_in_corpus", False),
            )
        )

    return GoldenSet(documents=documents, queries=queries)


def resolve_relevant_chunk_ids(
    labels: list[RelevanceLabel],
    *,
    chunks_by_document: dict[str, list[tuple[uuid.UUID, str]]],
) -> set[str]:
    """Turn (document, marker) labels into the chunk ids that satisfy them.

    Args:
        chunks_by_document: document key -> [(chunk_id, chunk_text)].

    Raises:
        ValueError: if a label matches no chunk. That means the marker text has
            drifted out of the corpus or the chunker split mid-marker, and
            silently scoring it as "nothing relevant exists" would understate
            recall for reasons that have nothing to do with the retriever.
    """
    resolved: set[str] = set()
    for label in labels:
        candidates = chunks_by_document.get(label.document, [])
        needle = label.contains.casefold()
        matches = [str(cid) for cid, text in candidates if needle in text.casefold()]
        if not matches:
            raise ValueError(
                f"Relevance label {label.document}/{label.contains!r} matched no chunk. "
                f"Either the marker no longer appears in the corpus, or chunking split "
                f"across it. Fix the label or the marker — do not let it score as a miss."
            )
        resolved.update(matches)
    return resolved
