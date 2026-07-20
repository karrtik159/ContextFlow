"""
CrewAI Tool — Knowledge Graph Search via Neo4j.

Allows CrewAI agents to traverse the Neo4j knowledge graph
to find entities and their relationships for multi-hop reasoning.

Uses the shared ``run_async()`` bridge for event-loop safety.
"""

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from agents.crews.tools.async_bridge import run_async


class GraphSearchInput(BaseModel):
    """Input schema for the graph search tool.

    Deliberately carries no ``user_id`` — see ``VectorSearchInput``.
    """

    entity_name: str = Field(description="Name of the entity to search for in the knowledge graph.")
    max_hops: int = Field(default=2, ge=1, le=3, description="Maximum relationship hops to traverse (1-3).")


class GraphSearchTool(BaseTool):
    name: str = "graph_search"
    description: str = (
        "Search the Neo4j knowledge graph for entities and their relationships. "
        "Traverses multi-hop connections to find related people, topics, "
        "preferences, and facts. Use this for understanding user context, "
        "preferences, and relationship-based reasoning. The traversal is "
        "automatically confined to the current user's subgraph."
    )
    args_schema: Type[BaseModel] = GraphSearchInput

    # Request-scoped tenant boundary, set by SupportCrew at construction.
    # Not part of args_schema: the LLM can neither read nor override it.
    user_id: str

    def _run(self, entity_name: str, max_hops: int = 2) -> str:
        """Traverse this user's subgraph for related entities."""
        from app.services.graph_search import find_related_entities

        try:
            results = run_async(
                find_related_entities(
                    entity_name,
                    user_id=self.user_id,
                    max_hops=min(max_hops, 3),
                )
            )
        except Exception as e:
            return f"Graph search error: {e}"

        if not results:
            return f"No entities related to '{entity_name}' found in the knowledge graph."

        output_lines = [f"Found {len(results)} relationships for '{entity_name}':\n"]
        for i, record in enumerate(results, 1):
            rels = " → ".join(record.get("relationships", []))
            related = record.get("related_entity", "unknown")
            output_lines.append(f"{i}. {entity_name} —[{rels}]→ {related}")

        return "\n".join(output_lines)

