"""
Synthesis port — the seam between retrieval and answer generation.

Phase 8. The knowledge path's final step was structurally welded to CrewAI:
`answer_knowledge_query` called the crew directly, so replacing the heaviest
dependency in the repo — the one whose opentelemetry pins force the voice
worker to be a separate uv project — meant editing the orchestration. Now the
orchestration calls `synthesize`, and the backend is a config value.

`SYNTHESIS_BACKEND="crewai"` is the default and the current behavior. The
"direct" backend exists so the Phase 9 flip is a measured config change, not a
rewrite; it must NOT be enabled before the RAGAS parity run, and the flip
deliberately changes `routed_to` (invariant 7) in Phase 9, not here.

The crewai implementation deliberately remains `rag_service._synthesize`:
its source carries the tenant-scoping contract that tests assert on
(`SupportCrew(user_id=user_id)`, never a kickoff input), and it is resolved
through the module at call time so existing test patches keep working.
"""

from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


async def synthesize(*, user_id: str, query: str, context: str) -> str:
    """Render the final answer from retrieved context, via the configured backend.

    Raises ValueError on an unknown backend — a typo in SYNTHESIS_BACKEND must
    fail loudly at first use, not silently pick a default that changes what
    model renders every answer.
    """
    backend = settings.SYNTHESIS_BACKEND

    if backend == "crewai":
        # Late import, attribute resolved at call time: rag_service._synthesize
        # is both the implementation and the established patch seam.
        from app.services import rag_service

        return await rag_service._synthesize(user_id=user_id, query=query, context=context)

    if backend == "direct":
        return await _synthesize_direct(query=query, context=context)

    raise ValueError(
        f"Unknown SYNTHESIS_BACKEND {backend!r} — expected 'crewai' or 'direct'."
    )


async def _synthesize_direct(*, query: str, context: str) -> str:
    """One chat-completion call, same system prompt the crew path uses.

    Routed through `direct_chat` (CLAUDE.md: all LLM access goes through the
    llm_provider builders). Sharing SYNTHESIS_SYSTEM_PROMPT with the crew path
    is deliberate twice over: parity is what Phase 9 measures, and the prompt
    is part of the semantic cache's `cache_version` fingerprint — two backends
    with different prompts would cross-serve each other's cached answers.
    """
    from app.services.llm_provider import direct_chat
    from app.services.rag_service import SYNTHESIS_SYSTEM_PROMPT

    user_content = f"{context}\n\nQuestion: {query}"
    return await direct_chat(user_content, system_prompt=SYNTHESIS_SYSTEM_PROMPT)
