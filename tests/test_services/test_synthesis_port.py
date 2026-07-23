"""
Synthesis port dispatch (Phase 8).

The port's whole job is to make the backend a config value while changing
nothing else. So: default lands on rag_service._synthesize (the established
patch seam — resolved at call time, so existing test patches keep working),
"direct" routes through llm_provider with the SAME system prompt, and a typo'd
backend fails loudly instead of silently choosing what model answers users.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.services.rag_service import SYNTHESIS_SYSTEM_PROMPT
from app.services.synthesis import synthesize


@pytest.mark.asyncio
async def test_default_backend_is_crewai():
    assert settings.SYNTHESIS_BACKEND == "crewai"


@pytest.mark.asyncio
async def test_crewai_backend_calls_rag_service_synthesize(monkeypatch):
    """The dispatch must resolve rag_service._synthesize at CALL time —
    that is what keeps `monkeypatch.setattr(rag_service, "_synthesize", ...)`
    in existing tests intercepting the crew path."""
    monkeypatch.setattr(settings, "SYNTHESIS_BACKEND", "crewai")
    seen: dict = {}

    async def fake_synthesize(*, user_id, query, context):
        seen.update(user_id=user_id, query=query, context=context)
        return "crew answer"

    monkeypatch.setattr("app.services.rag_service._synthesize", fake_synthesize)

    answer = await synthesize(user_id="u-1", query="q?", context="CTX")

    assert answer == "crew answer"
    assert seen == {"user_id": "u-1", "query": "q?", "context": "CTX"}


@pytest.mark.asyncio
async def test_direct_backend_routes_through_llm_provider(monkeypatch):
    monkeypatch.setattr(settings, "SYNTHESIS_BACKEND", "direct")
    seen: dict = {}

    async def fail_crew(**kwargs):
        raise AssertionError("direct backend must not touch the crew path")

    async def fake_direct_chat(query, *, system_prompt=None, history=None):
        seen["query"] = query
        seen["system_prompt"] = system_prompt
        return "direct answer"

    monkeypatch.setattr("app.services.rag_service._synthesize", fail_crew)
    monkeypatch.setattr("app.services.llm_provider.direct_chat", fake_direct_chat)

    answer = await synthesize(user_id="u-1", query="q?", context="CTX")

    assert answer == "direct answer"
    assert "CTX" in seen["query"] and "q?" in seen["query"]
    # Same prompt as the crew path — parity is what Phase 9 measures, and the
    # prompt is part of the semantic cache's version fingerprint.
    assert seen["system_prompt"] == SYNTHESIS_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_unknown_backend_fails_loudly(monkeypatch):
    monkeypatch.setattr(settings, "SYNTHESIS_BACKEND", "gpt-magic")
    with pytest.raises(ValueError, match="SYNTHESIS_BACKEND"):
        await synthesize(user_id="u", query="q", context="c")
