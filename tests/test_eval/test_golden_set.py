"""
Tests for the golden set and its relevance resolution.

The golden set is the yardstick, so it needs its own guard rails: a fixture that
silently stops matching would make every downstream metric wrong in the
flattering direction.
"""

from __future__ import annotations

import uuid

import pytest

from app.evals.golden_set import (
    DEFAULT_GOLDEN_SET,
    NOT_IN_CORPUS,
    GoldenQuery,
    RelevanceLabel,
    load_golden_set,
    resolve_relevant_chunk_ids,
)


@pytest.fixture(scope="module")
def golden():
    return load_golden_set(DEFAULT_GOLDEN_SET)


# ── Structural integrity ────────────────────────────────────

def test_golden_set_loads(golden):
    assert len(golden.documents) >= 5
    assert len(golden.queries) >= 10


def test_every_label_points_at_a_known_document(golden):
    keys = {d.key for d in golden.documents}
    for query in golden.queries:
        for label in query.relevant:
            assert label.document in keys


def test_every_marker_appears_in_its_document(golden):
    """The load-bearing invariant. A marker that has drifted out of the corpus
    resolves to zero chunks and would silently score as a retrieval miss."""
    by_key = {d.key: d.text.casefold() for d in golden.documents}
    for query in golden.queries:
        for label in query.relevant:
            assert label.contains.casefold() in by_key[label.document], (
                f"Query {query.id!r}: marker {label.contains!r} no longer appears in "
                f"document {label.document!r}."
            )


def test_query_ids_are_unique(golden):
    ids = [q.id for q in golden.queries]
    assert len(ids) == len(set(ids))


def test_set_contains_unanswerable_queries(golden):
    """Without these, a retriever that returns something for everything scores
    perfectly and the honest-empty path is never tested."""
    assert len(golden.unanswerable) >= 2
    for query in golden.unanswerable:
        assert query.reference == NOT_IN_CORPUS
        assert query.relevant == []


def test_answerable_and_unanswerable_partition_the_set(golden):
    assert len(golden.answerable) + len(golden.unanswerable) == len(golden.queries)


def test_set_contains_multi_document_queries(golden):
    """At least one query answerable from two documents, so fusion's reward for
    cross-source agreement is exercised rather than assumed."""
    assert any(len(q.relevant) > 1 for q in golden.answerable)


def test_domain_is_fictional(golden):
    """The corpus must not be answerable from model priors, or faithfulness
    scores measure prior knowledge instead of grounding."""
    for doc in golden.documents:
        assert "meridian" in doc.text.casefold()


# ── Relevance resolution ────────────────────────────────────

def test_resolve_matches_chunks_containing_the_marker():
    a, b = uuid.uuid4(), uuid.uuid4()
    resolved = resolve_relevant_chunk_ids(
        [RelevanceLabel(document="d", contains="rollback command")],
        chunks_by_document={"d": [(a, "Run the rollback command now."), (b, "Unrelated.")]},
    )
    assert resolved == {str(a)}


def test_resolve_is_case_insensitive():
    a = uuid.uuid4()
    resolved = resolve_relevant_chunk_ids(
        [RelevanceLabel(document="d", contains="ROLLBACK Command")],
        chunks_by_document={"d": [(a, "the rollback command")]},
    )
    assert resolved == {str(a)}


def test_resolve_can_match_several_chunks():
    a, b = uuid.uuid4(), uuid.uuid4()
    resolved = resolve_relevant_chunk_ids(
        [RelevanceLabel(document="d", contains="token")],
        chunks_by_document={"d": [(a, "a token"), (b, "another token")]},
    )
    assert resolved == {str(a), str(b)}


def test_resolve_unions_multiple_labels():
    a, b = uuid.uuid4(), uuid.uuid4()
    resolved = resolve_relevant_chunk_ids(
        [
            RelevanceLabel(document="d1", contains="alpha"),
            RelevanceLabel(document="d2", contains="beta"),
        ],
        chunks_by_document={"d1": [(a, "alpha here")], "d2": [(b, "beta here")]},
    )
    assert resolved == {str(a), str(b)}


def test_unmatched_label_raises_rather_than_scoring_as_a_miss():
    """Silently returning an empty set would understate recall for a reason
    that has nothing to do with the retriever."""
    with pytest.raises(ValueError, match="matched no chunk"):
        resolve_relevant_chunk_ids(
            [RelevanceLabel(document="d", contains="absent phrase")],
            chunks_by_document={"d": [(uuid.uuid4(), "something else")]},
        )


def test_unknown_document_raises():
    with pytest.raises(ValueError, match="matched no chunk"):
        resolve_relevant_chunk_ids(
            [RelevanceLabel(document="missing", contains="x")],
            chunks_by_document={"d": [(uuid.uuid4(), "x")]},
        )


# ── is_unanswerable ─────────────────────────────────────────

def test_unanswerable_detected_by_empty_labels():
    q = GoldenQuery(id="q", query="?", relevant=[], reference="anything")
    assert q.is_unanswerable


def test_unanswerable_detected_by_marker_reference():
    q = GoldenQuery(
        id="q", query="?", relevant=[RelevanceLabel("d", "x")], reference=NOT_IN_CORPUS
    )
    assert q.is_unanswerable


def test_answerable_query_is_not_flagged():
    q = GoldenQuery(id="q", query="?", relevant=[RelevanceLabel("d", "x")], reference="a")
    assert not q.is_unanswerable
