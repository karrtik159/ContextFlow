"""
Query rewriting — Phase 3.

The deterministic path is a pure function of the query string, so it is tested
exhaustively here rather than only end-to-end. The LLM-backed paths are tested
for the property that actually matters operationally: they are OFF by default
and they degrade to nothing when the model call fails, because a rewriting
failure must never fail a retrieval.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.services.retrieval.rewrite import (
    build_sparse_query,
    deterministic_rewrite,
    extract_identifiers,
    generate_hyde_document,
    generate_query_variants,
    rewrite_query,
)


class TestExtractIdentifiers:
    def test_release_tag_is_an_identifier(self):
        assert "20260714-af" in extract_identifiers("why did release 20260714-af fail")

    def test_acronyms_are_identifiers(self):
        found = extract_identifiers("how do I configure SAML SSO")
        assert "SAML" in found
        assert "SSO" in found

    def test_version_strings_survive(self):
        assert "v2.14.0" in extract_identifiers("what changed in v2.14.0")

    def test_dotted_settings_path(self):
        assert "hnsw.ef_search" in extract_identifiers("what does hnsw.ef_search do")

    def test_quoted_phrase_is_taken_literally(self):
        found = extract_identifiers('search for "Anchored support tier" please')
        assert "Anchored support tier" in found

    def test_ordinary_prose_yields_nothing(self):
        assert extract_identifiers("how long are audit logs kept") == []

    def test_deduplicates_case_insensitively(self):
        found = extract_identifiers("SAML and saml and SAML")
        assert len([f for f in found if f.casefold() == "saml"]) == 1

    def test_respects_limit(self):
        query = "E1042 E1043 E1044 E1045 E1046 E1047"
        assert len(extract_identifiers(query, limit=3)) == 3


class TestBuildSparseQuery:
    """The sparse query is the query text, unchanged.

    An earlier version appended identifiers back as double-quoted phrases to
    force an exact match. Measured against PostgreSQL 16, that produced a
    byte-identical tsquery: the consumer lexes with `to_tsvector`, to which
    double quotes are punctuation rather than phrase delimiters, and
    `tsvector_to_array` de-duplicates the appended copy away. These tests pin
    the honest behaviour so the no-op does not get reintroduced as if it worked.
    """

    def test_identifiers_do_not_alter_the_query(self):
        assert (
            build_sparse_query("rollback 20260714-af failed", ["20260714-af"])
            == "rollback 20260714-af failed"
        )

    def test_no_identifiers_passes_through_unchanged(self):
        assert build_sparse_query("how long are logs kept", []) == "how long are logs kept"

    def test_empty_query_stays_empty(self):
        assert build_sparse_query("   ", ["X"]) == ""

    def test_no_quote_characters_are_injected(self):
        """Injected quotes would be stripped by to_tsvector anyway, but they
        would also make the trace's sparse_query misrepresent what was run."""
        built = build_sparse_query('search "Anchored tier" now', ["Anchored tier"])
        assert built.count('"') == 2  # only the user's own quotes survive


class TestDeterministicRewrite:
    def test_dense_query_is_never_mutated(self):
        """Phase 2 established that aggressive normalization belongs on the
        cache key, not the retrieval path. Rewriting the dense query is a
        measurable change and must not happen by accident."""
        query = "please explain how do I roll back a bad deploy?"
        assert deterministic_rewrite(query).dense_query == query

    def test_costs_no_llm_call(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("deterministic_rewrite must not call an LLM")

        monkeypatch.setattr("app.services.llm_provider.get_async_llm_client", explode)
        assert deterministic_rewrite("what is the rate limit for a service token")

    def test_disabled_expansion_extracts_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "QUERY_EXPANSION_ENABLED", False)
        result = deterministic_rewrite("rollback 20260714-af")
        assert result.identifiers == []
        assert result.methods == []

    def test_dense_queries_is_just_the_query_by_default(self):
        assert deterministic_rewrite("hello there").dense_queries == ["hello there"]


class TestFlagGating:
    """HyDE and multi-query each cost an LLM call, which is the budget Phase 2
    freed by deleting the ReAct loop. They must stay off unless asked for."""

    def test_hyde_is_off_by_default(self):
        assert settings.HYDE_ENABLED is False

    def test_multi_query_is_off_by_default(self):
        assert settings.MULTI_QUERY_ENABLED is False

    async def test_hyde_returns_none_when_disabled_without_calling_llm(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("HYDE_ENABLED is False; no LLM call may happen")

        monkeypatch.setattr("app.services.llm_provider.get_async_llm_client", explode)
        assert await generate_hyde_document("anything") is None

    async def test_multi_query_returns_empty_when_disabled(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("MULTI_QUERY_ENABLED is False; no LLM call may happen")

        monkeypatch.setattr("app.services.llm_provider.get_async_llm_client", explode)
        assert await generate_query_variants("anything") == []


class TestLLMPathDegradation:
    async def test_hyde_failure_degrades_to_none(self, monkeypatch):
        monkeypatch.setattr(settings, "HYDE_ENABLED", True)

        async def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr("app.services.retrieval.rewrite._complete", boom)
        assert await generate_hyde_document("q") is None

    async def test_multi_query_failure_degrades_to_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", True)

        async def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr("app.services.retrieval.rewrite._complete", boom)
        assert await generate_query_variants("q") == []

    async def test_rewrite_query_survives_both_failing(self, monkeypatch):
        monkeypatch.setattr(settings, "HYDE_ENABLED", True)
        monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", True)

        async def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr("app.services.retrieval.rewrite._complete", boom)
        result = await rewrite_query("how do I roll back")
        assert result.dense_query == "how do I roll back"
        assert result.hyde_document is None
        assert result.variants == []

    async def test_multi_query_parses_and_dedups_lines(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", True)
        monkeypatch.setattr(settings, "MULTI_QUERY_COUNT", 3)

        async def fake(*args, **kwargs):
            # Numbering, bullets, a blank line, and an echo of the input — all
            # things a model actually emits despite being told not to.
            return "1. revert a deployment\n\n- undo a release\nq\nrevert a deployment"

        monkeypatch.setattr("app.services.retrieval.rewrite._complete", fake)
        variants = await generate_query_variants("q")
        assert variants == ["revert a deployment", "undo a release"]

    async def test_hyde_document_becomes_an_extra_dense_probe(self, monkeypatch):
        monkeypatch.setattr(settings, "HYDE_ENABLED", True)

        async def fake(*args, **kwargs):
            return "To roll back, run the rollback command with the previous tag."

        monkeypatch.setattr("app.services.retrieval.rewrite._complete", fake)
        result = await rewrite_query("how do I roll back")
        assert len(result.dense_queries) == 2
        assert result.dense_queries[0] == "how do I roll back"
        assert "hyde" in result.methods


@pytest.mark.parametrize(
    "query",
    ["", "   ", "?", "a", "SELECT * FROM chunks; --", "'''\"\"\"", "什么是速率限制"],
)
def test_rewrite_never_raises_on_hostile_input(query):
    """The rewrite stage runs before every retrieval; raising here would turn a
    weird query into a failed request."""
    result = deterministic_rewrite(query)
    assert isinstance(result.sparse_query, str)
