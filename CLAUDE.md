# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Package manager is `uv`. Python 3.11–3.12 for the backend (`requires-python = ">=3.11,<3.13"`).

```bash
# Backend
uv sync                                    # install main deps
uv sync --extra dev                        # + pytest, ruff, ragas
uv run uvicorn app.main:app --reload

# Tests
uv run pytest                              # asyncio_mode=auto, testpaths=["tests"]
uv run pytest tests/test_api/test_rag.py::test_rag_query_direct_path   # single test
uv run pytest tests/test_eval/             # RAGAS eval (needs OPENAI_API_KEY)
uv run python scripts/run_ragas_eval.py    # RAGAS against a *running* backend

# Lint (ruff: line-length 120, select E/F/I/UP)
uv run ruff check .
uv run ruff format .

# Migrations
uv run alembic revision --autogenerate -m "..."
uv run alembic upgrade head

# Voice worker — SEPARATE project, separate venv
cd agents/voice && uv sync && uv run python src/agent.py dev
cd agents/voice && uv run pytest

# Docker (only the `dev` and `voice-dev` profiles are active; prod services are commented out)
docker compose --profile dev --profile voice-dev up -d --build

# Gradio test harness
uv sync --extra demo && uv run python demo/gradio_app.py
```

Many tests under `tests/test_api/` build an `ASGITransport(app=app)` and therefore import the real app — they need PostgreSQL reachable. They mock crews/LLM/embeddings via `monkeypatch.setattr` on the *module path* (`"app.services.llm_provider.classify_intent"`), not on the importing module, because `app/api/v1/rag.py` imports those functions lazily inside the handler.

## Two isolated Python projects

`crewai` and `livekit-agents` pin conflicting `opentelemetry-sdk` versions. The voice worker is therefore a **standalone uv project** at `agents/voice/` with its own `pyproject.toml`/`uv.lock`, and it never imports `app.*`. It reaches the backend purely over HTTP (`agents/voice/src/rag_service.py` → `POST /api/v1/context/prefetch` and `/api/v1/rag/query`), authenticating with the `X-RAG-Service-Token` header.

Never add `livekit-agents` to the root `pyproject.toml`. The root only carries `livekit`/`livekit-api` for token minting.

## Request flow — `POST /api/v1/rag/query`

`app/api/v1/rag.py` is the system's centerpiece. Ordered pipeline:

1. `resolve_rag_user_id()` (`app/api/deps.py`) decides the authoritative user scope.
2. `sanitize_query()` (`app/services/cache_sanitizer.py`) — regex PII masking + filler stripping, producing a `normalized_query` used as the cache key.
3. `classify_intent()` — keyword heuristic first (0 ms), LLM fallback (~200 ms). Fails **open** to RAG.
4. Neo4j semantic cache lookup (authenticated users only) → `routed_to="cache"`.
5. Simple chat → `direct_chat`/`stream_direct_chat` → `routed_to="direct"`.
6. Knowledge query → `SupportCrew` via `asyncio.to_thread` → `routed_to="crewai"`.
7. Any crew failure or a suspiciously short answer degrades to `direct_chat` → `routed_to="direct_fallback"` (never a 500).
8. `BackgroundTasks` fire `MemoryCrew` and cache population.

`routed_to` is the observability contract; tests assert on it. Preserve those values when changing routing.

## Auth scoping rules (`resolve_rag_user_id`)

Three caller classes, deliberately asymmetric:
- **JWT user** — identity wins. A `user_id` in the body that disagrees with the token → 403.
- **Internal service token** (`X-RAG-Service-Token`, compared with `hmac.compare_digest`) — may supply any `user_id`.
- **Anonymous** — supplying a `user_id` → 403; **no `user_id` at all → 401**. `/rag/query` has no unscoped mode. Every store it reads (pgvector messages, the Neo4j subgraph, Mem0) is user-owned, so an unscoped run could only return nothing or read across tenants — it previously did the latter under a literal `"anonymous"` scope. The 401 also removes an unauthenticated path to a ~15-LLM-call request. Enforced by `test_rag_query_anonymous_is_rejected`, which asserts no crew, no classifier, and no embedding call happens.

## Embeddings are dimension-coupled

`settings.EMBEDDING_DIMENSIONS` is load-bearing in three places that must stay in sync:
- `Message.embedding` column type — `Vector(settings.EMBEDDING_DIMENSIONS)` in `app/models/message.py`, plus the `ix_messages_embedding_hnsw` HNSW/cosine index.
- The Neo4j `semantic_cache_index` vector index — `init_semantic_cache()` detects a dimension change at startup and **drops + recreates** the index.
- `_validate_embedding_dimensions()` raises if a provider returns a different width.

Switching provider (`openai` 1536 ↔ `huggingface`/MiniLM 384) requires an Alembic migration re-typing the column and re-embedding existing rows. `alembic/versions/d4e9f0a1b2c3_restore_message_embedding_dimension.py` exists precisely because of this.

`app/services/embeddings.py` is the single source of truth — semantic cache, pgvector search, CrewAI tools, and context prefetch all go through it. Do not construct OpenAI/SentenceTransformer clients elsewhere. Use `embed_text()` from sync/CrewAI threads, `embed_text_async()` from endpoints, `embed_text_async_safe()` for best-effort paths.

## Async/sync boundaries

- **CrewAI tools** (`agents/crews/tools/*.py`) run `_run()` synchronously in worker threads that may already own an event loop. Always use `run_async()` from `agents/crews/tools/async_bridge.py` — never `asyncio.run()`. It caches one loop per thread in `threading.local()`.
- `VectorSearchTool` keeps a **thread-local SQLAlchemy engine** (`pool_size=2`) rather than sharing the app engine, since it lives outside the request event loop.
- **Crew kickoff from FastAPI** must be wrapped in `asyncio.to_thread` — `Crew.kickoff()` is blocking.
- **Mem0 SDK** methods are blocking; wrap in `asyncio.to_thread` from async callers.

## Lazy, thread-safe singletons

The Neo4j driver (`app/services/graph_search.py`), the SentenceTransformer encoder (`app/services/embeddings.py`), and the Mem0 client (`app/memory/mem0_service.py`) all use double-checked locking and initialize on first use, **not** at import. `tests/test_agents/test_crew_config_split.py::test_graph_search_import_does_not_connect` enforces this so unit tests run without Neo4j. Preserve the pattern when adding external clients.

## Configuration

`app/core/config.py` composes ~15 single-concern `BaseSettings` classes into one `Settings` via multiple inheritance, exported as the module-level `settings` singleton. This is not cosmetic: `lifespan_factory` and `create_application` in `app/core/setup.py` branch on `isinstance(settings, RedisCacheSettings)`, `isinstance(settings, PostgresSettings)`, etc. Removing a base class silently disables its startup hook.

Secrets are `SecretStr` — always `.get_secret_value()`.

## Storage layout

- **PostgreSQL** — relational tables + `Message.embedding` pgvector column (HNSW/cosine). Managed by Alembic; `alembic/env.py` auto-imports every module under `app.models`.
- **Neo4j** — the knowledge graph *and* the semantic response cache (`:SemanticCache` nodes, `user_id`-scoped, exact-match then vector-similarity at threshold 0.95).
- **Mem0** — owns its own pgvector collection (`mem0_memories`) and Neo4j graph store. Alembic must not manage Mem0 tables.
- **Redis** — configured but the cache/rate-limit pools in `app/core/setup.py` are currently **no-op mocks**, and `rate_limiter_dependency` is a pass-through stub. Don't assume Redis-backed caching exists.

Note `create_tables_on_start=True` in `app/main.py` coexists with Alembic; `Base.metadata.create_all` runs on every startup.

## CrewAI crews

Two `@CrewBase` crews under `agents/crews/`, both `Process.sequential` with `memory=True` and `embedder=build_crewai_embedder()`:
- **SupportCrew** — `context_gatherer` (VectorSearchTool + GraphSearchTool + MemorySearchTool) → `answer_synthesizer` (**no tools**). 120 s cap.
- **MemoryCrew** — `entity_extractor` → `graph_updater` (MemoryStoreTool). 60 s cap, fire-and-forget background task.

`answer_synthesizer` deliberately holds no tools: it reads retrieved content, which is attacker-influenceable, so pairing it with a retrieval tool closed a stored-injection → exfiltration loop. It gets context via the task dependency on `retrieve_context`.

### Tenant scope is a constructor argument, never a prompt or tool argument

Both crews take `user_id` in `__init__` (`SupportCrew(user_id=...).crew().kickoff(inputs={"query": ...})`) and pass it into each tool at construction. Tools declare `user_id` as a **pydantic field on the tool, absent from `args_schema`**, so the LLM cannot read or override it.

This replaced a design where `user_id` was an LLM-filled tool argument whose intended value was interpolated into the task YAML — a tenant boundary that was only a suggestion to the model. Do not reintroduce `user_id` into any `args_schema`, and do not add `user_id` back to crew `kickoff` inputs. `tests/test_agents/test_tool_tenant_scoping.py` enforces both.

Correspondingly, `search_similar_messages()` and `find_related_entities()` **require** `user_id` and raise `ValueError` without it — the filter is not optional, because a conditional filter silently degrades into a cross-tenant scan.

Note: `@CrewBase` applies a metaclass that *rebuilds* the class, so a zero-arg `super().__init__()` in a crew `__init__` raises `TypeError`. Omit it; `CrewBaseMeta.__call__` runs its own initialization after your `__init__` returns (which is why `self.user_id` is set in time for the `@agent` methods).

Agent/task prompts live in per-crew YAML (`config/support_agents.yaml`, `config/support_tasks.yaml`, and the `memory_*` pair). The split is enforced by tests: each crew's tasks may only reference its own agents. Add prompt changes to YAML, not Python.

`build_crewai_llm()` / `get_async_llm_client()` in `app/services/llm_provider.py` abstract three providers (`openai` | `google` | `openrouter`) behind the OpenAI-compatible API; each has provider-specific model-name prefixing rules (`gemini/`, `openrouter/`). Route all LLM access through these builders.

## Skills

`.agents/skills/` holds vendored third-party skills pinned by `skills-lock.json` (postgres, rag-implementation, crewai-multi-agent, livekit-agents, pydantic, fastapi-pro, …). Treat them as read-only reference material — edits are overwritten on re-sync.
