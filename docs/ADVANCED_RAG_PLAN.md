# ContextFlow — Advanced RAG Migration Plan

**Status:** proposed, not started
**Author:** drafted 2026-07-20 from a three-dimension code review (retrieval quality, architecture, security/cost)
**Decisions locked with the project owner:**

| Decision | Choice |
|---|---|
| Corpus | Documents (new ingestion pipeline) **+** chat history as a secondary personalization signal |
| Query engine | **Deterministic** retrieval pipeline; LLM used only for final synthesis |
| Trace | **Persisted** stage-by-stage trace **and** inline citations in the API response |
| Anonymous access | `401` — `/rag/query` has no unscoped mode (already shipped, commit `3560bf9`) |

---

## 1. Where this actually stands

Be blunt about the starting point, because the plan only makes sense against it.

**There is no corpus.** The only embedded entity in the system is `Message.content` — whole chat messages, one vector each. A grep for `chunk|splitter|overlap` across `app/` and `agents/` returns nothing. What ships today is *conversational memory search* presented as retrieval-augmented generation. Every "advanced RAG" technique below is downstream of fixing that.

**Reciprocal Rank Fusion does not exist as code.** It appears twice — `agents/crews/config/support_tasks.yaml:9` and `support_agents.yaml:10` — as English prose instructing the model to perform it. Two of the three retrieval tools return prose with no scores at all, so the LLM is asked to execute a ranking algorithm over data it was never given. The output looks fused and is unfalsifiable.

**The retrieval path is an LLM ReAct loop.** `context_gatherer` has `max_iter=10` and three tools it calls one at a time, each gated on a model turn. Measured cost is roughly 7–15 sequential LLM round-trips and 3–5 redundant embedding calls per knowledge query. The 120 s `max_execution_time` is a realistic p99, not a safety margin.

**Nothing is retrieved with a relevance floor.** `search_similar_messages` is `ORDER BY ... LIMIT 5` with no threshold, and the tool renders the result as `"Found 5 relevant results:"` regardless. On an off-topic query the synthesizer receives five pieces of noise labelled relevant. This is the single largest hallucination driver in the system.

**Also absent:** reranking, hybrid/BM25, query rewriting, MMR, citations, groundedness checking, and any retrieval metric.

### Already fixed (commit `3560bf9`) — do not regress

- Tenant scope is a **crew constructor argument**, bound into each tool at construction. `user_id` is a pydantic field on the tool, **absent from `args_schema`**, so the LLM cannot address it.
- `search_similar_messages()` and `find_related_entities()` **require** `user_id` and raise `ValueError` without it.
- Graph traversal constrains **both** path endpoints.
- `answer_synthesizer` holds **no tools** (it reads attacker-influenceable content).
- `/rag/query` returns `401` before intent classification when no scope resolves.

`tests/test_agents/test_tool_tenant_scoping.py` guards these. Any new retrieval component must satisfy the same invariant: **scope is structural, never an LLM-supplied argument.**

---

## 2. Phase 0 — repair and delete before building

**Status: DONE.** Tests 127→130 passed / 2 skipped; ruff 82→59 errors. Note the
plan originally cited a 24-error ruff baseline — the measured baseline was 82.
Sanitizer resolved as **(b)**: `cache_sanitizer.py` → `query_normalizer.py`,
exposing a single `normalize_for_cache_key(str) -> str`. The masked-key collision
bug is *not* fixed here — it is documented in the module and pinned by
`test_masking_collides_distinct_queries`, which Phase 2 is expected to overturn.

Two regressions introduced by the `401` change, plus a verified dead-code sweep. Do this first; it is small and it stops the new work from being built on fiction.

### 2.1 Regressions to repair

| Item | Problem | Fix |
|---|---|---|
| `app/evals/ragas_framework.py:68-84` | `FastAPIRagTarget._request_json` sends `json=payload` and no auth header. `/rag/query` now returns `401`; supplying `"user_id": "eval-user"` anonymously returns `403`. **Every eval case fails.** | Send `X-RAG-Service-Token` (the header `deps.py:95` compares) from settings. |
| `tests/test_eval/fixtures/ragas_cases.json:47` | Reference answer asserts `answer_synthesizer with MemorySearchTool`. That tool was deliberately removed. The case now scores correct answers as wrong. | Update the reference text. |

### 2.2 Verified dead code — DELETE

Every item below was confirmed by reading the definition **and** grepping call sites.

| Target | Evidence |
|---|---|
| `app/api/v1/auth.py` (whole file) | Contains only a docstring. Zero importers; `router.py` never references it. |
| `app/core/setup.py:36-50` — 4 Redis pool functions | Each body is a single `logger.info("MOCK: ...")`. `import redis` appears **nowhere** in the repo. Drop the `redis[hiredis]` dependency too. |
| `app/core/setup.py:64-67,85` — `initialization_complete` | Constructed, `.set()` once, **never awaited or read**. |
| `app/api/deps.py:163-176` — `rate_limiter_dependency` | Body is `pass`. **Zero** call sites. (See Phase 6 — implement real limiting rather than keeping a stub that reads as coverage.) |
| `app/api/deps.py:144-159` — `get_current_superuser` / `CurrentSuperUser` | Zero call sites; no admin endpoints exist. |
| `app/core/security.py:125-127` — `decode_access_token` | Docstring claims "used by deps.py"; `deps.py` uses `verify_token`. Zero call sites. |
| `app/api/v1/rag.py:79` — `db: DBSession` | Never referenced in the handler. Opens a redundant Postgres session per request (ruff `ARG001`). |
| `config.py` — `DefaultRateLimitSettings`, `FirstUserSettings`, `LoggerSettings`, `RedisCacheSettings`, `RedisRateLimiterSettings` | 5 of 15 mixins; **9 settings with zero readers** (`DEFAULT_RATE_LIMIT_*`, `LOG_LEVEL`, `LOG_FORMAT_JSON`, `REDIS_*`, `ADMIN_*`). Remove the matching `isinstance` blocks in `setup.py`. |
| `setup.py` — the `isinstance(settings, ...)` dispatch | `Settings` inherits **all** mixins and `create_application` has exactly one call site with that singleton. Every guard is statically `True`. Inline the bodies; type the parameter as `Settings`. |
| Unused imports | `main.py:8` `FastAPI`, `main.py:9` `loguru.logger`, `users.py:7,10`, `test_async_bridge.py:12`. `ruff check --fix --select F401`. |
| `ragas_framework.py:50` `EvalCase.metadata`; `:84` `status_code` | Never read by any caller. |

**Pick one logging stack.** loguru appears in 2 files (one unused); stdlib `logging` in 7. Neither is configured — no `dictConfig`, no `logger.add()`, no `basicConfig`. Consolidate on stdlib and actually configure it (Phase 6).

### 2.3 The sanitizer — decide what it is

`sanitize_query()` has one production call site (`rag.py:120`) and only `normalized_query` is read, three times, all cache-key related. `requires_isolation`, `detected_pii_types`, and `original_query` are read **only by tests**.

Meanwhile the **raw** query flows to `classify_intent`, `direct_chat`, `stream_direct_chat`, the crew kickoff, and `_process_memory_background` → Mem0 → persisted. So a user's SSN is masked in the Neo4j cache key and sent verbatim to the LLM vendor and written to long-term memory. The module docstring calls this "sandboxing"; it is not.

There is also a correctness bug: the cache **key** is the masked text but the cached **answer** was generated from the raw text. Two distinct queries that mask identically (two different emails → both `[email]`) collide and serve each other's answers. Scoped per-user, so not cross-tenant — but wrong.

**Decide one:**
- **(a) It is a privacy control** → route `normalized_query` to every downstream consumer, and mask in `chat.py` before persisting. Note this changes answer quality, since masking is lossy.
- **(b) It is a cache-key normalizer** → delete the three dead fields, rename the module, and drop the security language from the docstring.

This plan assumes **(b)**, plus a separate real PII decision in Phase 6. Phase 2 splits normalization in two regardless (§4.1).

---

## 3. Target architecture

Aligned with the Microsoft/Azure AI Search advanced-RAG reference: *document cracking → chunking → enrichment → indexing*, then *rewrite → hybrid retrieve → fuse → rerank → threshold → grounded generate*, with evaluation closing the loop.

```
INGEST (offline, per document)
  crack → chunk (structure-first, token-aware) → enrich (heading path, metadata)
        → embed → index (HNSW cosine + GIN tsvector)

QUERY (online, deterministic until synthesis)
  normalize ─┬─ cache key (aggressive)
             └─ retrieval text (light)
  classify (cheap model / heuristic)
  rewrite   (expansion; optional HyDE, multi-query)
  retrieve  ── parallel ──┬─ dense  (pgvector, user-scoped)
                          ├─ sparse (tsvector BM25, user-scoped)
                          ├─ graph  (Neo4j, user-scoped subgraph)
                          └─ memory (Mem0, user-scoped)
  fuse      (real RRF in Python: Σ 1/(k + rank), k=60)
  rerank    (cross-encoder over top-N)
  threshold (relevance floor; empty ⇒ honest "no context")
  assemble  (fenced, numbered, citable blocks)
  generate  (ONE LLM call, cited)
  verify    (optional groundedness)
  persist   (trace + cache)
```

Every stage emits a record. The trace is the product, not a debug afterthought — it is what makes retrieval quality measurable instead of anecdotal.

### 3.1 Schema

```sql
documents
  id uuid pk, user_id uuid not null, source_uri text, title text,
  content_type text, checksum text, status text, token_count int,
  created_at timestamptz
  unique (user_id, checksum)          -- idempotent re-ingest

chunks
  id uuid pk,
  document_id uuid fk -> documents on delete cascade,
  user_id uuid not null,              -- DENORMALIZED, deliberately (see below)
  chunk_index int, text text,
  heading_path text,                  -- "Title > Section > Subsection"
  char_start int, char_end int, token_count int,
  embedding vector(EMBEDDING_DIMENSIONS),
  content_tsv tsvector generated always as (to_tsvector('english', text)) stored
  indexes:
    hnsw (embedding vector_cosine_ops)  with (m=16, ef_construction=64)
    gin  (content_tsv)
    btree (user_id, document_id)

retrieval_traces
  id uuid pk, user_id uuid not null, created_at timestamptz,
  original_query text, normalized_query text, rewritten_query text,
  routed_to text, total_latency_ms int,
  stages jsonb,                       -- [{name, latency_ms, in, out, meta}]
  final_chunk_ids uuid[], answer_hash text
  index: (user_id, created_at desc)
```

**Why `user_id` is denormalized onto `chunks`:** pgvector's HNSW scan walks `hnsw.ef_search` (default **40**) global candidates and applies the `WHERE` **after**. Joining to `documents` for the tenant filter means a user whose chunks aren't in the global top-40 gets **zero rows** — recall collapses toward 0 as the table grows. This is the same latent bug that exists today in `search_similar_messages`'s join to `chat_sessions`. Keeping `user_id` on the same row permits a partial-index or iterative-scan strategy. Set `hnsw.ef_search` explicitly per query (200+) and, on pgvector ≥ 0.8, enable `hnsw.iterative_scan = relaxed_order`.

**This is a tenant boundary.** `chunks.user_id` must be `NOT NULL` and every query must filter on it — same invariant as §1.

### 3.2 Chunking

Structure-first, token-aware, never naive character splitting:

1. Split on document structure (headings → paragraphs → sentences).
2. Pack sentences to a target token budget; never split mid-sentence.
3. Overlap adjacent chunks.
4. **Prepend `heading_path`** to the embedded text so an isolated chunk carries its context.

**The token budget is coupled to the embedding model and this is load-bearing.** Under `EMBEDDING_PROVIDER=huggingface`, MiniLM's `SentenceTransformer.encode` **silently truncates at 256 word-pieces** — no warning, no error. A 2,000-word chunk is represented by its first ~200 words. Under `openai`, the same input raises above 8,191 tokens and `embed_text_async_safe` swallows it into `embedding=None`, storing a permanently unsearchable row.

Add `EMBEDDING_MAX_TOKENS` to config alongside the existing `EMBEDDING_DIMENSIONS` coupling documented in CLAUDE.md, and derive chunk size from it (openai → ~512/64; MiniLM → ~256/32). Assert at ingestion that no chunk exceeds it; log loudly rather than truncating silently.

### 3.3 Contracts

```python
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID | None      # None for graph/memory hits
    text: str
    source: Literal["vector", "bm25", "graph", "memory"]
    rank: int                     # rank WITHIN its source list — RRF needs this
    score: float                  # raw source score, NOT comparable across sources
    heading_path: str | None
    citation_label: str           # "[1]" — stable through assembly

@dataclass
class StageRecord:
    name: str
    latency_ms: int
    input_summary: str
    output_summary: str
    metadata: dict                # k, thresholds, model, candidate counts

@dataclass
class RetrievalTrace:
    trace_id: UUID
    user_id: str
    stages: list[StageRecord]
    final_chunks: list[RetrievedChunk]
```

`rank` is mandatory on every source. RRF operates on ranks, not scores — the current tools emit neither, which is precisely why the YAML instruction can't work.

---

## 4. Phases

Each phase is independently shippable and independently reviewable. Do not batch them.

### Phase 1 — Corpus foundation

**Status: DONE.** Tests 130→189 passed / 2 skipped; ruff held at 76. Migration
`f1a2b3c4d5e6` verified against a throwaway `pgvector/pgvector:pg16` container:
upgrade, downgrade, and re-upgrade all clean, all three indexes created as
specified. E2E ingest verified with the real tiktoken counter.

Two corrections to this plan, both from empirical checks:
- The OpenAI ceiling is **8192**, not 8191 (verified in the installed `openai`
  package, not from memory). `EMBEDDING_MAX_TOKENS` is set accordingly.
- **Token counts are not additive.** The first packing implementation summed
  per-unit counts and blew the ceiling on CJK, where `count(a) + count(b)` is
  far below `count(a + b)`. Packing now measures the assembled candidate chunk.
  Any future retrieval code that budgets by summing has the same bug.

MiniLM's silent truncation is CONFIRMED, and worse than the plan stated: a
3000-word input embeds identically to its first 200 words (cosine 1.0000001)
with no warning, because `SentenceTransformer.encode()` passes
`truncation=True` internally and suppresses the tokenizer's own message.
- `documents` + `chunks` models; Alembic migration (HNSW + GIN + composite indexes).
- `app/services/chunking.py` — structure-first token-aware splitter. **Pure function, heavily unit-tested**: headings, code blocks, tables, CJK, a single 50k-token paragraph, empty input, token-budget boundaries.
- `EMBEDDING_MAX_TOKENS` in config; assert on ingest.
- `POST /api/v1/documents` — upload/ingest (user-scoped), idempotent by `(user_id, checksum)`.
- Batch embedding (do **not** embed one chunk per HTTP call).
- **Exit criteria:** a document can be ingested, chunked, embedded, and retrieved by direct SQL, with `chunks.user_id` populated and enforced.

### Phase 2 — Deterministic pipeline + trace

**Status: DONE.** Tests 189→225 passed / 1 skipped; ruff 76→74. Migration
`a7b8c9d0e1f2` verified against a throwaway container (upgrade, downgrade,
re-upgrade). Full pipeline verified E2E with real pgvector: correct chunk ranked
first, cross-tenant retrieval returned 0 rows, off-topic query rejected by the
floor, all 7 stages persisted and JSONB-queryable.

Deviations and findings:
- **`RETRIEVAL_MIN_SIMILARITY = 0.25` is an unvalidated guess.** It is the one
  number here with no empirical basis, and it is the difference between the
  honest-empty path and answering from noise. Phase 5 must calibrate it against
  a labelled set; it is also provider-specific.
- The **sparse/BM25 arm stays in Phase 3** as the plan assigns it, so the
  fan-out is dense-corpus + dense-messages + graph + memory. The GIN index from
  Phase 1 is already in place for it.
- **Fusion identity must be source-independent.** The first implementation
  qualified id-less items by source, which meant the same fact from two arms
  never fused — reducing RRF to a weighted concatenation and removing exactly
  the cross-source agreement it exists to reward.
- The three retrieval tools were **deleted**, not left dangling: their only
  consumer was `context_gatherer`. `MemoryStoreTool` remains the sole LLM-facing
  tool. Tenant guards were retargeted onto the retrieval sources.
- `Crew(memory=False)` — CrewAI's own memory re-embeds and re-retrieves on every
  kickoff, which would defeat the one-LLM-call goal and duplicate work the
  pipeline just did deterministically.
- `app/services/rag_pipeline.py` — stage framework emitting `StageRecord`, returning `RetrievalTrace`.
- Split normalization: aggressive **cache-key** normalization vs light **retrieval** normalization. Drop the filler-stripping regexes from the retrieval path (`^please explain\s+` etc. change meaning and modern encoders don't need them).
- Parallel fan-out via `asyncio.gather` over dense/sparse/graph/memory.
- **Real RRF in Python.** Delete the RRF language from both YAMLs in the same commit.
- Relevance threshold with an honest empty path.
- Replace the `context_gatherer` ReAct loop. `retrieve_context` as an agent task goes away.
- `retrieval_traces` table + persistence.
- **Extract `rag_query` into the service layer.** Today it is a 174-line handler with ~13 decision points, a nested closure, a nested async generator, and a nested sync function — the `routed_to` contract cannot be unit-tested without ASGI + PostgreSQL. After extraction the handler is ~20 lines and the routing matrix is a pure-function test.
- **Exit criteria:** a knowledge query completes with 1 LLM call (synthesis) instead of 7–15; every stage appears in the trace; `routed_to` values preserved.

### Phase 3 — Rerank + rewrite

**Status: DONE.** Tests 269→355 passed / 1 skipped; ruff held at 73. No
migration — the sparse arm rides entirely on the GIN index Phase 1 already
created (`alembic check`: no new upgrade operations). Measured against real
pgvector with `all-MiniLM-L6-v2` and `BAAI/bge-reranker-base`, no API spend.

**Measured, k=3, 16 documents / 52 chunks, 43 answerable queries, floor 0.25:**

| configuration        | recall@3 | nDCG@3 | MRR   | prec@3 | misses |
|----------------------|----------|--------|-------|--------|--------|
| dense only (Phase 2) | 0.988    | 0.974  | 0.977 | 0.364  | 0      |
| + sparse             | 0.988    | 0.943  | 0.938 | 0.364  | 0      |
| + rerank             | 1.000    | 0.982  | 0.977 | 0.372  | 0      |
| **+ both (Phase 3)** | **1.000**| **0.982**| 0.977| 0.372  | 0      |

The exit criterion is met, but the honest reading is that the margin is small
and the ablation matters more than the headline. Three findings:

**Finding 1 — the sparse arm is only safe downstream of the reranker.**
On its own it *costs* nDCG (0.974 → 0.943): OR-semantics injects loosely-related
chunks that fusion cannot discriminate. Its value is recall into the candidate
pool, which the cross-encoder then sorts out. Enabling `SPARSE_ENABLED` with
`RERANK_ENABLED=False` is a measured regression, not a partial improvement.

**Finding 2 — the cross-encoder does NOT solve abstention.** Phase 5 concluded
that no cosine floor both preserves recall and rejects topically-near
unanswerable queries, and predicted the reranker would supply that signal
because it scores relevance directly. It does not. Abstention is still **0/5**,
and the unanswerable "Which OIDC claims does Meridian map to workspace roles?"
scored **0.992** — higher than most genuinely correct answers. A cross-encoder
scores *aboutness*, and an unanswerable question about a documented subject is
maximally about it.

The `--sweep-rerank` calibration confirms there is no usable operating point:

| RERANK_MIN_SCORE | recall@3 | nDCG@3 | MRR   | misses | abstain |
|------------------|----------|--------|-------|--------|---------|
| 0.00             | 1.000    | 0.982  | 0.977 | 0      | 0/5     |
| 0.01             | 0.988    | 0.975  | 0.977 | 0      | 1/5     |
| 0.05             | 0.977    | 0.961  | 0.953 | 1      | 1/5     |
| 0.10             | 0.953    | 0.938  | 0.930 | 2      | 1/5     |
| 0.20             | 0.907    | 0.891  | 0.884 | 4      | 1/5     |
| 0.30             | 0.884    | 0.868  | 0.860 | 5      | 1/5     |
| 0.50             | 0.860    | 0.845  | 0.837 | 6      | 1/5     |
| 0.70             | 0.860    | 0.845  | 0.837 | 6      | 1/5     |

Abstention buys exactly **one** of five unanswerable queries, at 0.01, and then
never improves no matter how much recall is spent — by 0.30 recall has fallen
to 0.884 with five queries retrieving nothing relevant, still abstaining on
only that same one. The cross-encoder is not a weak abstention signal; it is
close to a non-signal. `RERANK_MIN_SCORE` therefore ships at **0.0 (floor
disabled)**, which is the correct default rather than a cautious one: every
non-zero value measured is a strict loss.

**Abstention needs a different mechanism, and the literature names it.** Google
/ UC San Diego, *Sufficient Context: A New Lens on RAG Systems* (ICLR 2025,
arXiv 2411.06037), draws exactly the distinction this measurement ran into:
context is **sufficient** if it contains all the information needed for a
definitive answer, and sufficiency is not relevance — context can be maximally
relevant and still insufficient. They also report the failure mode seen here:
RAG *reduces* a model's willingness to abstain. Their autorater (prompted LLM,
binary sufficient/insufficient over query + retrieved chunks, run BEFORE
generation so it works as a filter rather than a post-hoc audit) reaches ~93%
against human expert labels, and pairing it with self-rated confidence improves
selective accuracy by up to 10 points over confidence alone. A local
entailment/NLI classifier (TRUE-NLI style; Vectara HHEM-2.1-Open is a
`flan-t5-base` cross-encoder that loads through the same `sentence_transformers`
CrossEncoder path as the reranker) is the cheaper approximation, reported
slightly below the LLM autorater. This is Phase 4 work — see §Phase 4.

**Finding 3 — three stages read as working while doing nothing.** Each was
caught only by measurement, and each looked correct in review:
- The **sparse arm returned zero rows on almost every query.**
  `websearch_to_tsquery` (and `plainto_tsquery`) AND every term, so "How often
  do log files rotate?" became `often & log & file & rotat`. The arm ran, cost
  a round trip, and contributed nothing — visible only because `+ sparse`
  scored *identical to three decimals* to dense-only. Now ORs lexemes.
- The **reranker made retrieval worse** (recall 0.988 → 0.953) because it
  scored chunk text *without its heading path*. "Synchronisation runs every
  four hours…" never says "directory", so the model read it as a strong answer
  to "How often are encryption keys rotated?". `build_context_block` already
  showed the heading path to the synthesis model; scoring on less context than
  the reader gets is a plain mismatch.
- **Identifier phrase-quoting was a no-op.** `build_sparse_query` wrapped
  identifiers in double quotes for `websearch_to_tsquery` — reasoning that
  stopped applying once the consumer changed. `to_tsvector` is a *document*
  parser: quotes are punctuation, and `tsvector_to_array` de-duplicates the
  appended copy. Verified the executed tsquery is byte-identical with and
  without it. Removed.

**Two defects fixed that predate this phase:**
- **Concurrent `AsyncSession` use.** Phase 2's fan-out ran the dense and
  message arms in one `asyncio.gather` over a shared session. SQLAlchemy
  rejects this (`InvalidRequestError: concurrent operations are not
  permitted`, verified against the installed version); Phase 3 would have made
  it three arms. DB arms now run serially, with Neo4j and Mem0 — the arms
  actually worth overlapping — still concurrent with the whole group.
- **Transaction poisoning.** PostgreSQL aborts the transaction on any
  statement error, so one failing arm made every subsequent arm fail with
  `InFailedSQLTransactionError`. Because `_run_arm` swallows arm failures, the
  cascade was silent: the trace showed several independent "arm failed"
  entries whose stated causes were all the same downstream symptom. Each DB arm
  now runs in a `SAVEPOINT`, which clears the error without discarding work the
  request did earlier. (The transaction did *not* go on to fail the request's
  final `commit()` — tested.)

**Golden set expanded 6 → 16 documents, 18 → 48 queries** to satisfy Phase 5's
Finding 2. The additions are near-miss distractors, not more subject matter:
four documents now describe a "rotation" (token, encryption key, on-call shift,
log file); "retained for N" appears with six different values of N; quota (a
limit at rest) and rate limit (a limit on speed) each explicitly disclaim the
other; a CLI error-code reference supplies identifier queries the dense arm is
structurally bad at.

**Not verified:** `RERANK_MIN_SCORE` calibration (`--sweep-rerank`), HyDE and
multi-query (implemented, flag-gated **off**, never measured — they cost an LLM
call each and must earn it against the golden set before being enabled), and
the reranker's behaviour with `text-embedding-3-small`, whose candidate pool
differs from MiniLM's.

- Cross-encoder reranker (`BAAI/bge-reranker-base`, ~30 ms for 25 pairs on CPU) behind the same lazy thread-safe singleton pattern as the SentenceTransformer encoder — **must not connect or load at import time** (`test_graph_search_import_does_not_connect` enforces this discipline).
- Retrieve wide (k≈25), rerank to 5. Today k=5 *is* the final context, so a relevant chunk ranked 6th is unrecoverable.
- Hybrid sparse retrieval via `ts_rank_cd` over `content_tsv`. Dense-only retrieval fails on exact identifiers, error codes, and rare proper nouns — precisely the "factual lookup" queries the classifier routes here.
- Query rewriting/expansion; optional HyDE and multi-query behind flags.
- **Exit criteria:** measurable recall@k and nDCG improvement on a labelled set (Phase 5 provides it).

### Phase 4 — Grounding + citations
- Chunks reach the prompt as **fenced, numbered, citable blocks**; retrieved text is delimited and explicitly marked untrusted.
- Response gains `citations: [{chunk_id, document_id, heading_path, score, span}]` and `trace_id`.
- Post-hoc validation that every `[n]` in the answer resolves to a real supplied chunk; reject/repair otherwise.
- Optional groundedness verification pass.
- **Cache only grounded answers.** Today `populate_semantic_cache` fires whenever the crew returns ≥5 characters, so a hallucinated or tool-error-contaminated answer is cached and served to that user indefinitely. Gate on a `grounded` flag; never cache when a retrieval stage errored.
- **Semantic cache hardening:** add TTL (`c.timestamp` is written and never read for expiry), and include a `cache_version` derived from `hash(EMBEDDING_MODEL + LLM_MODEL + prompt_version)` in the MERGE key so a model or prompt change invalidates. Raise the `0.95` threshold and make it provider-specific — on short normalized queries, encoders routinely exceed 0.95 for pairs differing only by a negation or a single entity ("is X covered" vs "is X **not** covered").
- **Exit criteria:** every non-cached answer carries resolvable citations.

### Phase 5 — Evaluation

**Status: DONE** (run before Phase 3 deliberately, so the reranker is measured
rather than assumed). Tests 225→269 passed / 1 skipped; ruff held at 74.
Measured for real with `all-MiniLM-L6-v2` against pgvector — no API spend.

**Measured, k=3, 19 chunks, 16 answerable queries, floor 0.25:**
recall@3 1.000 · precision@3 0.375 · MRR 1.000 · nDCG@3 1.000 · misses 0.
At k=1: recall 0.938, MRR 1.000 — the top hit is relevant for every query.

**Finding 1 — the similarity floor is inert, and no single value fixes it.**
`RETRIEVAL_MIN_SIMILARITY` was 0.25 by guesswork. Measured, anything ≤0.35 is a
no-op for MiniLM: the unanswerable "configure SAML SSO" query still retrieved 3
chunks at similarity 0.467. Abstention was **0/2** at the shipped default. The
sweep shows 0.40 → 1/2 abstention at recall 0.969; 0.50 → 2/2 at recall 0.844.
There is no floor that achieves both. Topically-near-but-unanswerable queries
outscore genuinely relevant chunks, so a cosine threshold cannot separate them.
**This is the measured case for Phase 3's cross-encoder reranker**, which scores
query-document relevance directly rather than embedding proximity. The default
is left at 0.25 and annotated: it remains uncalibrated for
`text-embedding-3-small`, whose distribution differs.

**Finding 2 — the golden set validates retrieval but cannot compare
strategies.** MRR 1.000 at every k on a 19-chunk corpus means the set is too
small and too easy to discriminate. It will catch a regression; it will *not*
show that a reranker helps. Phase 3 needs a larger, harder corpus with more
near-miss distractors before its exit criterion ("measurable recall@k and nDCG
improvement") can be met honestly.

**Not verified:** the RAGAS generation half is wired (`--generation`) but never
executed — it needs a running backend and a judge LLM. Faithfulness, answer
relevance, and context precision/recall are unmeasured.
- Fix `FastAPIRagTarget` auth and the stale fixture (Phase 0 if not already done).
- **Retrieval** metrics from the trace: recall@k, MRR, nDCG — these are what Phases 1–3 are optimizing and are currently unmeasurable.
- **Generation** metrics via RAGAS: faithfulness, answer relevance, context precision/recall.
- A small labelled golden set committed to the repo.
- Wire into CI. **There is no `.github/` directory today** — nothing runs automatically.
- **Exit criteria:** a single command reports retrieval + generation metrics; regressions are visible.

### Phase 6 — Operational readiness
- **Real rate limiting.** Currently zero: the stub has no call sites and no Redis exists. Authenticated users can still burn unbounded spend.
- **Correlation IDs** — no request-ID middleware exists; logs identify requests by `query[:80]`, and nothing ties a log line to its LangSmith trace or its background `MemoryCrew` run. Return it in a response header.
- **Metrics** — counters on `routed_to`, cache hit rate, `direct_fallback` rate, per-stage latency histograms, tokens/cost per tenant. Alert on `direct_fallback` rate: a total CrewAI outage currently looks like a healthy service returning slightly worse answers.
- **Fail-loud telemetry** — `telemetry.py` returns early with an `info` log when `LANGSMITH_API_KEY` is unset. A production deploy with a missing env var runs fully untraced and silent.
- `/health/ready` with real dependency probes (`/health` returns `{"status":"healthy"}` unconditionally).
- **Gate `MemoryCrew` on answer substance** — it fires on *every* query including trivial chat, costing 3–6 LLM calls to remember "thanks".
- Cheap `CLASSIFIER_MODEL`; lower `max_iter`; explicit per-request token ceiling.
- Security leftovers: CORS `["*"]` + `allow_credentials=True` (Starlette echoes the Origin, defeating the browser protection); default `SECRET_KEY` boots and mints forgeable tokens; `str(e)` leaked to clients in `context.py`/`memory.py`; invalid JWT silently degrading to anonymous instead of `401`.
- Decide the real PII policy (§2.3).

---

## 5. Invariants — must hold at every phase

1. **Tenant scope is structural.** Never an LLM-supplied tool argument, never interpolated into prompt text as an instruction to obey. Bound at construction; absent from `args_schema`.
2. **Retrieval functions require `user_id`** and raise without it. A conditional tenant filter silently degrades into a cross-tenant scan — that is exactly how the original bug shipped.
3. **Fail closed.** An unparseable or missing scope returns nothing; it never widens.
4. **Retrieved content is data, not instructions.** Fenced and marked untrusted wherever it enters a prompt.
5. **No agent that reads retrieved content also holds a tenant-parameterized retrieval tool.**
6. **No silent truncation.** Over-long input is chunked or logged loudly, never quietly cut.
7. **`routed_to` is the observability contract.** Preserve existing values; additions are deliberate and documented.
8. **External clients initialize lazily and thread-safely** (double-checked locking), never at import.
9. **Don't cache what isn't grounded.**

---

## 6. Carry-forward prompt

Paste into a fresh session, or hand to a subagent. Replace the phase marker.

```
You are continuing an advanced-RAG migration on the ContextFlow repo
(d:\Work\sample_demo\Learning\ContextFlow). A FastAPI + CrewAI + pgvector +
Neo4j + Mem0 stack.

READ FIRST, before writing any code:
  - docs/ADVANCED_RAG_PLAN.md  (this plan — current state, architecture, phases)
  - CLAUDE.md                  (repo conventions, async/sync boundaries,
                                embedding dimension coupling, config composition)

TASK: implement **Phase <N>** of the plan. Only that phase. Do not batch phases.

GROUND RULES
- Verify before you assert. This codebase contains code that claims to do things
  it does not — "Reciprocal Rank Fusion" exists only as prose in a YAML prompt;
  a "rate limiter" has a `pass` body and zero call sites; Redis pool functions
  are `logger.info("MOCK: ...")`. Read the implementation and grep the call
  sites; never trust a docstring, a name, or a comment.
- The invariants in §5 of the plan are non-negotiable. Tenant scope is
  structural — bound to the object at construction, absent from any
  LLM-facing schema. `tests/test_agents/test_tool_tenant_scoping.py` enforces
  this; if a change makes it fail, the change is wrong, not the test.
- Retrieval functions require `user_id` and raise without it. Never reintroduce
  a conditional tenant filter.
- Retrieved document text is untrusted data. Fence it; never let it reach a
  prompt as instructions.
- Async boundaries: CrewAI tools use `run_async()` from
  `agents/crews/tools/async_bridge.py`, never `asyncio.run()`. Crew kickoff
  from FastAPI is wrapped in `asyncio.to_thread`. Mem0 SDK calls are blocking.
- Embeddings go through `app/services/embeddings.py` only. Do not construct
  OpenAI or SentenceTransformer clients anywhere else.
- `settings.EMBEDDING_DIMENSIONS` is coupled to the pgvector column type, the
  Neo4j `semantic_cache_index`, and a runtime width check. Changing the
  provider requires a migration and re-embedding.
- New external clients follow the lazy double-checked-locking singleton
  pattern; they must not connect or load a model at import time.
- Prompts live in per-crew YAML, not Python.

VERIFICATION — required, not optional
- `uv run pytest` must pass. Report the actual counts.
- `uv run ruff check .` must not add errors above the existing baseline
  (24 pre-existing at the time of writing — measure, don't assume).
- Add tests for the behaviour you build, including a regression guard for any
  invariant the phase touches.
- Empirically verify anything you inferred about a third-party library rather
  than assuming it. Concrete precedent: `@CrewBase` applies a metaclass that
  *rebuilds* the class, so a zero-arg `super().__init__()` inside a crew
  `__init__` raises TypeError. That was only caught by instantiating it.
- If you cannot verify something (e.g. a Mem0 Neo4j property name without a
  live graph), make the code **fail closed**, say so explicitly in your report,
  and flag it for confirmation. Do not present an assumption as a fact.

REPORT
State what you changed, what you verified and how, what you could not verify,
and anything you found that contradicts this plan. If a phase assumption turns
out to be wrong, say so rather than working around it silently.
```

---

## 7. Sequencing notes

- **Phase 0 and Phase 1 are the unlock.** Reranking, hybrid search, and RRF are all meaningless without a chunked corpus to rank.
- **Phase 2 pays for the rest.** Collapsing 7–15 LLM calls into 1 is both the latency and the cost win, and the trace it produces is what makes Phases 3 and 5 measurable.
- **Do not skip Phase 5.** Every technique in Phase 3 is a hypothesis until measured. Adding a reranker without recall@k is cargo-culting.
- Phase 6 is independent of 1–5 and can proceed in parallel if someone else picks it up — with one exception: rate limiting should not wait, since the `401` closed only the *anonymous* cost path.
