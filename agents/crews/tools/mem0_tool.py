"""
CrewAI Tool — Mem0 memory writes.

MemoryStoreTool is the only remaining LLM-facing tool in the system. The three
retrieval tools (vector, graph, memory search) were deleted in Phase 2: their
sole consumer was SupportCrew's ``context_gatherer``, and retrieval now runs
deterministically in ``app/services/retrieval/`` with no model in the call path.

That makes the tenant boundary strictly stronger than it was. Scope used to be
a pydantic field the LLM could not *address*; for retrieval it is now not
exposed to the LLM at all, because the LLM no longer performs retrieval.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MemoryStoreInput(BaseModel):
    """Input schema for storing a memory.

    Deliberately carries no ``user_id``. It previously did, and the intended
    value was interpolated into the task prompt for the agent to copy across —
    which made the tenant boundary a *suggestion in prompt text* rather than an
    enforced control. Any injected instruction in retrieved content could
    substitute another user's id; for a write tool that meant cross-tenant
    memory *poisoning*. Scope now lives on the tool instance.
    """

    content: str = Field(description="The memory content to store (fact, preference, entity).")


class MemoryStoreTool(BaseTool):
    name: str = "memory_store"
    description: str = (
        "Store a new fact, preference, or entity into the user's long-term memory. "
        "Use this after extracting important information from conversations "
        "that should be remembered across sessions."
    )
    args_schema: type[BaseModel] = MemoryStoreInput

    # Request-scoped tenant boundary, set by the crew at construction.
    # Not part of args_schema: the LLM can neither read nor override it.
    user_id: str

    def _run(self, content: str) -> str:
        """Store a memory for the current user via Mem0."""
        from app.memory.mem0_service import Mem0Service, clean_memory_content

        clean_content = clean_memory_content(content)
        if clean_content is None:
            return "Skipped memory store: extracted content was too low-signal to persist."

        try:
            Mem0Service.add_memory(clean_content, user_id=self.user_id)
            return f"Memory stored successfully: {clean_content[:100]}"
        except Exception as e:
            return f"Memory store error: {e}"
