"""
Regression guards for the CrewAI tool tenant boundary.

Context — all three retrieval tools previously leaked across tenants:

  * ``VectorSearchTool`` called ``search_similar_messages`` without a
    ``user_id``, and that filter was applied conditionally, so the query
    degraded into a global scan over every user's messages.
  * ``find_related_entities`` had no ``user_id`` predicate at all.
  * ``MemorySearchTool`` / ``MemoryStoreTool`` took ``user_id`` as an
    **LLM-filled argument**, with the intended value interpolated into task
    prompt text. That makes the tenant boundary a suggestion the model is
    asked to honour, which prompt-injected content in retrieved documents
    can override — for the store tool, that meant cross-tenant writes.

The invariant these tests protect: **tenant scope is bound to the tool
instance at construction and is never reachable from the LLM-facing schema.**
"""

import pytest

from agents.crews.tools.graph_search_tool import GraphSearchInput, GraphSearchTool
from agents.crews.tools.mem0_tool import (
    MemorySearchInput,
    MemorySearchTool,
    MemoryStoreInput,
    MemoryStoreTool,
)
from agents.crews.tools.vector_search_tool import VectorSearchInput, VectorSearchTool

# (schema, tool class) for every tenant-scoped tool.
SCOPED_TOOLS = [
    (VectorSearchInput, VectorSearchTool),
    (GraphSearchInput, GraphSearchTool),
    (MemorySearchInput, MemorySearchTool),
    (MemoryStoreInput, MemoryStoreTool),
]


@pytest.mark.parametrize(
    "schema,tool_cls",
    SCOPED_TOOLS,
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_user_id_is_not_llm_addressable(schema, tool_cls):
    """The LLM-facing args schema must not expose user_id.

    If this fails, an injected instruction in retrieved content can name a
    tenant and the agent will comply.
    """
    assert "user_id" not in schema.model_fields, (
        f"{schema.__name__} exposes 'user_id' to the LLM. Tenant scope must "
        f"live on the tool instance, not in the argument schema."
    )


@pytest.mark.parametrize(
    "schema,tool_cls",
    SCOPED_TOOLS,
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_user_id_is_a_required_constructor_field(schema, tool_cls):
    """Constructing a scoped tool without a user_id must fail loudly."""
    assert "user_id" in tool_cls.model_fields
    with pytest.raises(Exception):
        tool_cls()


def test_vector_search_rejects_missing_user_id():
    """The service function must refuse an unscoped search outright.

    Defence in depth: even if a future caller forgets to pass a scope, the
    query must not silently become a cross-tenant scan.
    """
    import asyncio

    from app.services.vector_search import search_similar_messages

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(
            search_similar_messages(db=None, query_embedding=[0.1], user_id=None)
        )


def test_graph_search_rejects_missing_user_id():
    """Graph traversal must refuse to run unscoped."""
    import asyncio

    from app.services.graph_search import find_related_entities

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(find_related_entities("Alice", user_id=""))


def test_graph_search_cypher_scopes_both_path_endpoints():
    """Both ends of a traversed path must carry the user_id predicate.

    Scoping only the start node would still let a path walk out into another
    tenant's subgraph and return their entity names.
    """
    import inspect

    from app.services import graph_search

    source = inspect.getsource(graph_search.find_related_entities)
    # Start node and end node each constrained by $user_id.
    assert source.count("user_id: $user_id") >= 2, (
        "find_related_entities must constrain both the start and end node "
        "by user_id, otherwise traversal escapes the tenant subgraph."
    )


def test_vector_search_tool_fails_closed_on_unparseable_scope():
    """A malformed scope must not fall through to an unscoped query."""
    tool = VectorSearchTool(user_id="not-a-uuid")
    result = tool._run(query="anything")
    assert "unavailable" in result.lower()
