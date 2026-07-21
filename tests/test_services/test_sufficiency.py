"""
Context sufficiency — the abstention gate.

Phase 3 measured that abstention cannot be recovered from a relevance score:
neither RETRIEVAL_MIN_SIMILARITY nor RERANK_MIN_SCORE separates answerable from
unanswerable queries at any threshold. Sufficiency is a different quantity
(arXiv 2411.06037) and gets its own signal.

Roughly half of this file pins what the term-coverage detector CANNOT do. That
is deliberate: its blind spots are structural, not bugs, and a test suite that
only demonstrates the wins would make the mechanism look general when it is
narrow. The categories come from UAEval4RAG (arXiv 2412.12300).
"""

from __future__ import annotations

import uuid

import pytest

from app.core.config import settings
from app.services.retrieval.contracts import RetrievedChunk
from app.services.retrieval.sufficiency import (
    assess_sufficiency,
    salient_terms,
)


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source="vector", rank=1, score=0.9, chunk_id=uuid.uuid4()
    )


CORPUS = [
    _chunk(
        "Meridian rotates the data encryption key for each workspace every ninety "
        "days. Rotation is automatic and requires no action from an account owner."
    ),
    _chunk(
        "A workspace token is scoped to a single workspace and expires after ninety "
        "days. It can be rotated without downtime by issuing the replacement before "
        "revoking the original."
    ),
]


class TestSalientTerms:
    def test_drops_function_words(self):
        assert salient_terms("How do I rotate a token?") == ["rotate", "token"]

    def test_deduplicates(self):
        assert salient_terms("token token TOKEN").count("token") == 1

    def test_keeps_identifiers(self):
        assert "20260714-af" in salient_terms("roll back 20260714-af")

    def test_drops_very_short_tokens(self):
        assert "ab" not in salient_terms("ab token")

    def test_empty_query_is_empty(self):
        assert salient_terms("") == []


def _gate(monkeypatch, uncovered, *, max_uncovered=0):
    """Point the gate at a fake corpus lookup returning `uncovered`."""
    monkeypatch.setattr(settings, "SUFFICIENCY_ENABLED", True)
    monkeypatch.setattr(settings, "SUFFICIENCY_MAX_UNCOVERED_TERMS", max_uncovered)

    async def fake(db, *, terms, user_id, ts_config=None):
        return [t for t in uncovered if t in terms]

    monkeypatch.setattr("app.services.retrieval.sources.find_uncovered_terms", fake)


class TestStructuralBlindSpots:
    """What corpus term-coverage CANNOT detect, by construction.

    Every query here is built entirely from words the corpus contains, so
    `find_uncovered_terms` returns nothing and the gate approves. These are not
    failing tests — they pin the boundary of the mechanism so it is not
    mistaken for a general answerability check, and they are why the golden set
    labels such queries `terms_all_in_corpus` and reports them separately.

    Categories are from UAEval4RAG (arXiv 2412.12300).
    """

    async def test_underspecified_query_is_approved(self, monkeypatch):
        """'How long is it retained?' — the corpus answers this several times
        with different values. Coverage sees only familiar words."""
        _gate(monkeypatch, uncovered=[])
        verdict = await assess_sufficiency(None, "How long is it retained?", CORPUS, user_id=1)
        assert verdict.sufficient is True

    async def test_false_presupposition_is_approved(self, monkeypatch):
        _gate(monkeypatch, uncovered=[])
        verdict = await assess_sufficiency(
            None, "How do I extend the fourteen day artifact window?", CORPUS, user_id=1
        )
        assert verdict.sufficient is True

    async def test_nonsensical_query_is_approved(self, monkeypatch):
        _gate(monkeypatch, uncovered=[])
        verdict = await assess_sufficiency(
            None, "How do I rotate a webhook into Wednesday?", CORPUS, user_id=1
        )
        assert verdict.sufficient is True


class TestTolerance:
    async def test_max_uncovered_allows_slack(self, monkeypatch):
        """Strict 0 refuses a real question over one unusual word.

        Measured against the real corpus, "configure" does not appear anywhere
        in it, so "How do I configure X?" abstains at max_uncovered=0 even
        though X is fully documented. That false-abstention rate, not the
        abstention rate, is what decides whether this gate can ship.
        """
        query = "How do I configure SAML rotation?"
        _gate(monkeypatch, uncovered=["configure", "saml"], max_uncovered=0)
        assert (await assess_sufficiency(None, query, CORPUS, user_id=1)).sufficient is False

        _gate(monkeypatch, uncovered=["configure"], max_uncovered=0)
        assert (await assess_sufficiency(None, query, CORPUS, user_id=1)).sufficient is False

        _gate(monkeypatch, uncovered=["configure"], max_uncovered=1)
        assert (await assess_sufficiency(None, query, CORPUS, user_id=1)).sufficient is True


class TestGate:
    """The gate now checks the CORPUS, not the retrieved chunks, so these use
    a fake `find_uncovered_terms`. The corpus query itself is verified against
    real PostgreSQL in the eval harness."""

    async def test_disabled_gate_always_returns_sufficient(self, monkeypatch):
        """Callers must never branch on the flag themselves, and the trace must
        always carry a verdict."""
        monkeypatch.setattr(settings, "SUFFICIENCY_ENABLED", False)
        verdict = await assess_sufficiency(None, "absent nonsense xyzzy", [], user_id=None)
        assert verdict.sufficient is True
        assert verdict.method == "disabled"

    async def test_enabled_gate_abstains_on_uncovered_term(self, monkeypatch):
        monkeypatch.setattr(settings, "SUFFICIENCY_ENABLED", True)
        monkeypatch.setattr(settings, "SUFFICIENCY_MAX_UNCOVERED_TERMS", 0)

        async def fake(db, *, terms, user_id, ts_config=None):
            return ["saml"]

        monkeypatch.setattr(
            "app.services.retrieval.sources.find_uncovered_terms", fake
        )
        verdict = await assess_sufficiency(None, "How do I configure SAML?", CORPUS, user_id=1)
        assert verdict.sufficient is False
        assert verdict.uncovered_terms == ["saml"]

    async def test_covered_query_is_sufficient(self, monkeypatch):
        monkeypatch.setattr(settings, "SUFFICIENCY_ENABLED", True)

        async def fake(db, *, terms, user_id, ts_config=None):
            return []

        monkeypatch.setattr(
            "app.services.retrieval.sources.find_uncovered_terms", fake
        )
        verdict = await assess_sufficiency(None, "how is the key rotated", CORPUS, user_id=1)
        assert verdict.sufficient is True

    async def test_empty_context_abstains(self, monkeypatch):
        monkeypatch.setattr(settings, "SUFFICIENCY_ENABLED", True)
        verdict = await assess_sufficiency(None, "anything", [], user_id=1)
        assert verdict.sufficient is False

    async def test_db_failure_defaults_to_sufficient(self, monkeypatch):
        """This gate exists to suppress bad answers. It must never become a new
        way to lose good ones."""
        monkeypatch.setattr(settings, "SUFFICIENCY_ENABLED", True)

        async def boom(db, *, terms, user_id, ts_config=None):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(
            "app.services.retrieval.sources.find_uncovered_terms", boom
        )
        verdict = await assess_sufficiency(None, "how is the key rotated", CORPUS, user_id=1)
        assert verdict.sufficient is True
        assert "check failed" in verdict.reason

    def test_is_off_by_default(self):
        """Enabling this can only ADD refusals, and wrongly refusing a real
        question is worse than answering a bad one."""
        assert settings.SUFFICIENCY_ENABLED is False



@pytest.mark.parametrize(
    "query", ["", "   ", "?", "a", "'''", "什么是速率限制", "SELECT * FROM chunks; --"]
)
async def test_never_raises_on_hostile_input(query, monkeypatch):
    _gate(monkeypatch, uncovered=[])
    verdict = await assess_sufficiency(None, query, CORPUS, user_id=1)
    assert isinstance(verdict.sufficient, bool)
