"""
CrewAI Tool — Mem0 Memory Operations.

Allows CrewAI agents to store and retrieve long-term user memories
via the Mem0 service (backed by pgvector + Neo4j).
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MemorySearchInput(BaseModel):
    """Input schema for memory search.

    Deliberately carries no ``user_id``. It previously did, and the intended
    value was interpolated into the task prompt for the agent to copy across —
    which made the tenant boundary a *suggestion in prompt text* rather than
    an enforced control. Any injected instruction in retrieved content could
    substitute another user's id. Scope now lives on the tool instance.
    """

    query: str = Field(description="Search query to find relevant user memories.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of memories to return.")


class MemoryStoreInput(BaseModel):
    """Input schema for storing a memory.

    Deliberately carries no ``user_id`` — see ``MemorySearchInput``. For a
    write tool the same defect allowed cross-tenant memory *poisoning*.
    """

    content: str = Field(description="The memory content to store (fact, preference, entity).")


class MemorySearchTool(BaseTool):
    name: str = "memory_search"
    description: str = (
        "Search the long-term memory store for the current user's past "
        "preferences, facts, and conversation history. Returns the most "
        "relevant memories for personalization. Use this to understand user "
        "context before answering. The search is automatically confined to "
        "the current user — you cannot search another user's memories."
    )
    args_schema: type[BaseModel] = MemorySearchInput

    # Request-scoped tenant boundary, set by the crew at construction.
    # Not part of args_schema: the LLM can neither read nor override it.
    user_id: str

    def _run(self, query: str, limit: int = 5) -> str:
        """Search Mem0 for this user's relevant memories."""
        from app.memory.mem0_service import Mem0Service

        try:
            # Route through the service classmethod, not get_client(), so the
            # query normalization and empty-query guard apply.
            results = Mem0Service.search_memories(query, user_id=self.user_id, limit=limit)
        except Exception as e:
            return f"Memory search error: {e}"

        if not results:
            return f"No memories found matching '{query}'."

        output_lines = [f"Found {len(results)} memories:\n"]
        for i, mem in enumerate(results, 1):
            text = mem.get("memory", mem.get("text", str(mem)))
            score = mem.get("score", "N/A")
            output_lines.append(f"{i}. [score={score}] {text}")

        return "\n".join(output_lines)


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
