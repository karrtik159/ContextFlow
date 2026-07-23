"""
Citation validation (Phase 4 exit criterion).

`grounded` is earned, not assumed: an answer is grounded iff it cites at least
one supplied block and every citation resolves. Unresolvable labels are
stripped (the user never sees a dangling footnote) and the answer becomes
uncacheable — a confabulated citation is positive evidence the model was not
reading its context.
"""

from __future__ import annotations

import uuid

import pytest

from app.services import rag_service
from app.services.grounding import validate_citations
from app.services.rag_service import answer_knowledge_query
from app.services.retrieval.contracts import RetrievalTrace, RetrievedChunk

USER = str(uuid.uuid4())


def _chunk(label: str, text: str = "supporting evidence") -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source="vector", rank=int(label.strip("[]")), score=0.9,
        chunk_id=uuid.uuid4(), citation_label=label,
    )


# ── validate_citations ──────────────────────────────────────


def test_fully_resolvable_answer_is_grounded():
    check = validate_citations(
        "Log files rotate every four hours [1], configurable per workspace [2].",
        [_chunk("[1]"), _chunk("[2]")],
    )
    assert check.grounded is True
    assert check.was_repaired is False
    assert check.cited == ("[1]", "[2]")
    assert check.answer.endswith("[2].")


def test_unresolvable_citation_is_stripped_and_ungrounded():
    check = validate_citations(
        "Rotation happens every four hours [1], and backups run nightly [7].",
        [_chunk("[1]")],
    )
    assert check.grounded is False
    assert check.unresolvable == ("[7]",)
    assert "[7]" not in check.answer
    assert "[1]" in check.answer, "resolvable citations must survive the repair"
    assert "nightly." in check.answer, "stripping must not leave a stranded space"


def test_answer_with_no_citations_is_ungrounded_but_untouched():
    answer = "The context does not contain information about SAML SSO."
    check = validate_citations(answer, [_chunk("[1]")])
    assert check.grounded is False
    assert check.answer == answer


def test_no_supplied_chunks_makes_any_citation_unresolvable():
    check = validate_citations("Fact [1].", [])
    assert check.grounded is False
    assert check.unresolvable == ("[1]",)
    assert check.answer == "Fact."


def test_repeated_citations_count_once():
    check = validate_citations("A [1]. B [1]. C [2].", [_chunk("[1]"), _chunk("[2]")])
    assert check.cited == ("[1]", "[2]")
    assert check.grounded is True


def test_multi_digit_labels_resolve():
    check = validate_citations("See [12].", [_chunk("[12]")])
    assert check.grounded is True


# ── The wiring: grounded flows into cacheability ────────────


@pytest.fixture
def patched(monkeypatch):
    state: dict = {"answer": "cited answer [1]."}

    async def fake_run_retrieval(db, **kwargs):
        trace = RetrievalTrace(
            trace_id=uuid.uuid4(), user_id=USER, original_query="q", normalized_query="q",
        )
        trace.final_chunks = [_chunk("[1]")]
        return trace

    async def fake_synthesize(*, user_id, query, context):
        return state["answer"]

    monkeypatch.setattr(rag_service, "run_retrieval", fake_run_retrieval)
    monkeypatch.setattr(rag_service, "_synthesize", fake_synthesize)
    return state


async def _run():
    return await answer_knowledge_query(
        None, user_id=USER, original_query="What is X?",
        retrieval_query="What is X?", query_embedding=[0.1],
    )


async def test_cited_answer_is_grounded_and_cacheable(patched):
    outcome = await _run()
    assert outcome.grounded is True
    assert outcome.cacheable is True
    cite_stage = next(s for s in outcome.trace.stages if s.name == "cite_check")
    assert cite_stage.metadata["grounded"] is True


async def test_uncited_answer_is_returned_but_never_cached(patched):
    patched["answer"] = "An answer that cites nothing at all, in plain prose."
    outcome = await _run()
    assert outcome.routed_to == "crewai"
    assert outcome.answer == patched["answer"]
    assert outcome.grounded is False
    assert outcome.cacheable is False


async def test_confabulated_citation_is_repaired_and_uncacheable(patched):
    patched["answer"] = "Rotation is hourly [1], and the retention is 30 days [9]."
    outcome = await _run()
    assert "[9]" not in outcome.answer
    assert "[1]" in outcome.answer
    assert outcome.grounded is False
    assert outcome.cacheable is False
    cite_stage = next(s for s in outcome.trace.stages if s.name == "cite_check")
    assert cite_stage.metadata["unresolvable"] == ["[9]"]
    assert cite_stage.metadata["repaired"] is True


async def test_answer_that_was_only_bad_citations_degrades(monkeypatch, patched):
    """Nothing left after repair is the short-answer failure, and takes the
    same direct_fallback path."""
    calls = {"direct": 0}

    async def fake_direct_chat(query, **kwargs):
        calls["direct"] += 1
        return "fallback answer"

    monkeypatch.setattr("app.services.llm_provider.direct_chat", fake_direct_chat)
    patched["answer"] = "[7] [8]"

    outcome = await _run()

    assert outcome.routed_to == "direct_fallback"
    assert outcome.grounded is False
    assert calls["direct"] == 1
