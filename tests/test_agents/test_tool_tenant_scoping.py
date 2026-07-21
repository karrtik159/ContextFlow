"""
Regression guards for the tenant boundary.

Context — retrieval previously leaked across tenants three separate ways:

  * ``VectorSearchTool`` called ``search_similar_messages`` without a
    ``user_id``, and that filter was applied conditionally, so the query
    degraded into a global scan over every user's messages.
  * ``find_related_entities`` had no ``user_id`` predicate at all.
  * ``MemorySearchTool`` / ``MemoryStoreTool`` took ``user_id`` as an
    **LLM-filled argument**, with the intended value interpolated into task
    prompt text. That makes the tenant boundary a suggestion the model is asked
    to honour, which prompt-injected content in retrieved documents can
    override — for the store tool, that meant cross-tenant writes.

The invariant: **tenant scope is structural — bound to the object at
construction or passed as a required argument, and never reachable from an
LLM-facing schema.**

Phase 2 moved retrieval out of the agent loop entirely, so the three retrieval
tools are gone. Scope for retrieval is now a required function argument with no
model in the call path, which is strictly stronger than a non-addressable
pydantic field. These tests were retargeted accordingly: the surviving tool is
covered as before, and equivalent guards now cover the retrieval sources that
replaced the deleted tools.
"""

import asyncio
import inspect
import uuid

import pytest

from agents.crews.tools.mem0_tool import MemoryStoreInput, MemoryStoreTool

# (schema, tool class) for every tenant-scoped LLM-facing tool that remains.
SCOPED_TOOLS = [
    (MemoryStoreInput, MemoryStoreTool),
]


@pytest.mark.parametrize(
    "schema,tool_cls", SCOPED_TOOLS, ids=lambda v: getattr(v, "__name__", str(v))
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
    "schema,tool_cls", SCOPED_TOOLS, ids=lambda v: getattr(v, "__name__", str(v))
)
def test_user_id_is_a_required_constructor_field(schema, tool_cls):
    """Constructing a scoped tool without a user_id must fail loudly."""
    assert "user_id" in tool_cls.model_fields
    with pytest.raises(Exception):
        tool_cls()


def test_deleted_retrieval_tools_are_not_reintroduced():
    """The retrieval tools were removed because retrieval left the agent loop.

    Reintroducing one would put a tenant-parameterized retrieval tool back in
    the hands of an agent that reads attacker-influenceable content — the exact
    stored-injection -> exfiltration loop that removing them closed (§5.5).
    """
    import agents.crews.tools as tools_pkg

    module_dir = tools_pkg.__path__[0]
    import os

    present = {f for f in os.listdir(module_dir) if f.endswith(".py")}
    for forbidden in {"vector_search_tool.py", "graph_search_tool.py"}:
        assert forbidden not in present, (
            f"{forbidden} was deleted in Phase 2. Retrieval is deterministic and "
            f"must not be reachable from an agent."
        )


def test_synthesizer_holds_no_tools():
    """The synthesizer reads retrieved content, so it must hold no tools."""
    source = inspect.getsource(
        __import__("agents.crews.support_crew", fromlist=["SupportCrew"])
    )
    assert "tools=[]" in source, "answer_synthesizer must be constructed with no tools"


# ── Retrieval sources — scope is a required argument ────────

def test_search_chunks_rejects_missing_user_id():
    from app.services.retrieval.sources import search_chunks

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(search_chunks(None, query_embedding=[0.1], user_id=None))


def test_search_chunks_sparse_rejects_missing_user_id():
    from app.services.retrieval.sources import search_chunks_sparse

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(search_chunks_sparse(None, query="anything", user_id=None))


def test_search_messages_rejects_missing_user_id():
    from app.services.retrieval.sources import search_messages

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(search_messages(None, query_embedding=[0.1], user_id=None))


def test_search_graph_rejects_missing_user_id():
    from app.services.retrieval.sources import search_graph

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(search_graph(query="anything", user_id=""))


def test_search_memory_rejects_missing_user_id():
    from app.services.retrieval.sources import search_memory

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(search_memory(query="anything", user_id=""))


def test_run_retrieval_rejects_missing_user_id():
    from app.services.retrieval.pipeline import run_retrieval

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(
            run_retrieval(
                None,
                user_id="",
                original_query="q",
                retrieval_query="q",
                query_embedding=[0.1],
            )
        )


def test_run_retrieval_fails_closed_on_unparseable_scope():
    """A malformed scope returns nothing; it never widens into an unscoped run."""
    from app.services.retrieval.pipeline import run_retrieval

    trace = asyncio.run(
        run_retrieval(
            None,
            user_id="not-a-uuid",
            original_query="q",
            retrieval_query="q",
            query_embedding=[0.1],
        )
    )
    assert trace.final_chunks == []
    assert not trace.has_context
    assert any(s.metadata.get("failed_closed") for s in trace.stages)


def test_chunk_query_filters_on_chunks_user_id_directly():
    """The dense arm must filter on chunks.user_id, not via a join to documents.

    An HNSW scan applies the WHERE clause after walking its candidate list, so a
    tenant filter on a joined table silently collapses recall as the table
    grows. This is a correctness property, not a performance nicety.
    """
    from app.services.retrieval import sources

    source = inspect.getsource(sources.search_chunks)
    assert "Chunk.user_id == user_id" in source
    assert "Document" not in source, (
        "search_chunks must not reach documents for the tenant filter"
    )


def test_sparse_query_filters_on_chunks_user_id_directly():
    """The Phase 3 sparse arm carries the same tenant rule as the dense one.

    A new retrieval arm is a new place for the original cross-tenant defect to
    reappear, and a GIN scan has no HNSW-style excuse to hide behind — an
    unscoped `@@` match would simply return every tenant's chunks.
    """
    from app.services.retrieval import sources

    source = inspect.getsource(sources.search_chunks_sparse)
    assert "Chunk.user_id == user_id" in source
    assert "Document" not in source


def test_sparse_query_does_not_interpolate_the_ts_config_into_sql():
    """The text-search config reaches Postgres as a bound parameter.

    It is settings-derived rather than user-derived today, but an f-string here
    is one refactor away from being a user-derived one.
    """
    from app.services.retrieval import sources

    source = inspect.getsource(sources.search_chunks_sparse)
    assert "bindparam" in source
    assert 'f"' not in source.split("tsquery =")[-1].split("stmt =")[0]


def _code_without_docstring(fn) -> str:
    """Source of `fn` with its docstring removed.

    These guards assert on what the code DOES. A docstring that explains why a
    rejected alternative was rejected must not read as the code using it —
    otherwise documenting a decision breaks the test that enforces it.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]
    return ast.unparse(tree)


def test_sparse_arm_uses_or_semantics_not_and():
    """The sparse arm must OR its terms, not AND them.

    `websearch_to_tsquery` and `plainto_tsquery` both conjoin every term, so
    "How often do log files rotate?" becomes `often & log & file & rotat` and
    matches only a chunk containing all four. Measured against the golden
    corpus that returned ZERO rows for almost every query: the arm ran, cost a
    round trip, and contributed nothing to fusion. Restoring either function
    here would silently make the arm decorative again.
    """
    from app.services.retrieval import sources

    body = _code_without_docstring(sources.search_chunks_sparse)
    assert "websearch_to_tsquery" not in body
    assert "plainto_tsquery" not in body
    assert "tsvector_to_array" in body
    # ast.unparse normalizes string quoting, so match on the separator itself.
    assert "string_agg" in body
    assert " | " in body, "lexemes must be joined with the tsquery OR operator"


def test_sparse_arm_quotes_lexemes_before_reparsing():
    """Each lexeme is `quote_literal`-wrapped on the way back into to_tsquery.

    Without it a hyphenated identifier such as `20260714-af` is re-parsed as a
    phrase expression rather than matched as a lexeme.
    """
    from app.services.retrieval import sources

    source = inspect.getsource(sources.search_chunks_sparse)
    assert "quote_literal" in source


# ── Legacy service-level guards, still enforced ─────────────

def test_graph_search_rejects_missing_user_id():
    from app.services.graph_search import find_related_entities

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(find_related_entities("Alice", user_id=""))


def test_graph_search_cypher_scopes_both_path_endpoints():
    """Both ends of a traversed path must carry the user_id predicate.

    Scoping only the start node would still let a path walk out into another
    tenant's subgraph and return their entity names.
    """
    from app.services import graph_search

    source = inspect.getsource(graph_search.find_related_entities)
    assert source.count("user_id: $user_id") >= 2, (
        "find_related_entities must constrain both the start and end node "
        "by user_id, otherwise traversal escapes the tenant subgraph."
    )


def test_vector_search_rejects_missing_user_id():
    from app.services.vector_search import search_similar_messages

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(search_similar_messages(db=None, query_embedding=[0.1], user_id=None))


def test_support_crew_requires_user_id():
    from agents.crews.support_crew import SupportCrew

    with pytest.raises(ValueError, match="user_id"):
        SupportCrew(user_id="")


def test_support_crew_does_not_take_user_id_as_a_kickoff_input():
    """Scope is a constructor argument. Putting it back into kickoff inputs
    would interpolate it into prompt text for the model to pass along."""
    from app.services import rag_service

    source = inspect.getsource(rag_service._synthesize)
    assert 'SupportCrew(user_id=user_id)' in source
    assert '"user_id"' not in source.split("kickoff")[-1], (
        "user_id must not appear in kickoff inputs"
    )


def test_retrieval_sources_are_not_llm_tools():
    """No retrieval function may be exposed to an agent as a tool.

    Checks the module's actual objects rather than its text, so the prose
    explaining why there is no args_schema does not trip the assertion.
    """
    from crewai.tools import BaseTool

    from app.services.retrieval import sources

    for name in dir(sources):
        obj = getattr(sources, name)
        if isinstance(obj, type) and issubclass(obj, BaseTool):
            raise AssertionError(f"{name} exposes retrieval to an agent as a tool")
        assert not hasattr(obj, "args_schema"), f"{name} looks like an LLM-facing tool"


def test_uuid_scope_is_used_for_chunk_queries():
    """Sanity: the sources module converts scope to UUID rather than
    interpolating a string into SQL."""
    from app.services.retrieval.sources import _require_user_id

    scope = uuid.uuid4()
    assert _require_user_id(scope, "fn") == scope
    with pytest.raises(ValueError):
        _require_user_id(None, "fn")
