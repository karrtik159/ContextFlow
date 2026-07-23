"""
Core configuration — modular Pydantic BaseSettings grouped by concern.

Each sub-settings class handles one domain (app, auth, DB, Redis, etc.).
The final `Settings` class composes them all via multiple inheritance,
loading values from .env automatically.

Usage:
    from app.core.config import settings
    print(settings.POSTGRES_URI)
"""

import os
from enum import Enum

from pydantic import SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Application ─────────────────────────────────────────────
class AppSettings(BaseSettings):
    APP_NAME: str = "ContextFlow"
    APP_DESCRIPTION: str | None = "Real-time Voice & Deep Memory AI"
    APP_VERSION: str | None = "0.1.0"
    LICENSE_NAME: str | None = "MIT"
    CONTACT_NAME: str | None = None
    CONTACT_EMAIL: str | None = None


# ── Environment ─────────────────────────────────────────────
class EnvironmentOption(str, Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentSettings(BaseSettings):
    ENVIRONMENT: EnvironmentOption = EnvironmentOption.LOCAL


# ── Auth / JWT ──────────────────────────────────────────────
class CryptSettings(BaseSettings):
    SECRET_KEY: SecretStr = SecretStr("super-secret-change-me-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7


# ── PostgreSQL ──────────────────────────────────────────────
class PostgresSettings(BaseSettings):
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "changeme"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "contextflow_d"
    POSTGRES_ASYNC_PREFIX: str = "postgresql+asyncpg://"
    POSTGRES_SYNC_PREFIX: str = "postgresql://"
    POSTGRES_URL: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def POSTGRES_URI(self) -> str:
        """Builds user:pass@host:port/db fragment."""
        credentials = f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
        location = f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        return f"{credentials}@{location}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Full async connection URL for SQLAlchemy / Alembic."""
        if self.POSTGRES_URL:
            return self.POSTGRES_URL
        return f"{self.POSTGRES_ASYNC_PREFIX}{self.POSTGRES_URI}"


# ── Neo4j ───────────────────────────────────────────────────
class Neo4jSettings(BaseSettings):
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: SecretStr = SecretStr("changeme")


# NOTE: Redis and rate-limit settings previously lived here. Nothing read them —
# no Redis client is installed and the rate limiter was a `pass` stub. They are
# removed rather than kept as configuration that implies capability the service
# does not have. Real rate limiting is Phase 6 of docs/ADVANCED_RAG_PLAN.md and
# should reintroduce only the settings it actually consumes.


# ── CORS ────────────────────────────────────────────────────
class CORSSettings(BaseSettings):
    CORS_ORIGINS: list[str] = ["*"]
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]


# NOTE: FirstUserSettings (ADMIN_*) is removed — there is no seeding routine and
# no admin endpoint. It shipped a default admin password that nothing consumed.


# ── AI Services ─────────────────────────────────────────────
class AISettings(BaseSettings):
    # LLM Provider: "openai" | "google" | "openrouter"
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4.1-mini"

    # Model for the intent classifier's LLM fallback. Empty → LLM_MODEL. The
    # classifier emits ONE word; paying the flagship model's rates for it is
    # pure waste — point this at the cheapest model the provider offers.
    CLASSIFIER_MODEL: str = ""

    # Per-request completion ceiling for direct chat and direct synthesis.
    # A runaway generation is a cost bug the caller cannot see; the prompt
    # already asks for ~150 words, so this is a backstop, not a constraint.
    LLM_MAX_TOKENS: int = 1024

    # OpenAI / OpenRouter
    OPENAI_API_KEY: SecretStr = SecretStr("")
    OPENAI_BASE_URL: str = ""

    # Google Gemini
    GOOGLE_API_KEY: SecretStr = SecretStr("")

    # Embeddings
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    HUGGINGFACE_API_KEY: SecretStr = SecretStr("")

    # Hard input ceiling of the embedding model, in tokens. Coupled to
    # EMBEDDING_MODEL exactly as EMBEDDING_DIMENSIONS is — switching provider
    # requires updating both. Chunk sizes derive from this; ingestion refuses to
    # store a chunk that exceeds it rather than letting the encoder truncate.
    #
    #   openai/text-embedding-3-*    → 8192, per the installed openai package
    #     ("8192 tokens for all embedding models",
    #      openai/resources/embeddings.py). Verified in-package, not from memory.
    #   huggingface/all-MiniLM-L6-v2 → 256 word-pieces, INCLUDING [CLS]/[SEP].
    #     Measured: max_seq_length == 256, and SentenceTransformer.encode()
    #     truncates past it with no warning, no error, and no exception — a
    #     3000-word input embeds identically to its first 200 words
    #     (cosine 1.0000001). This is why ingestion asserts instead of trusting
    #     the encoder to complain.
    EMBEDDING_MAX_TOKENS: int = 8192

    # Target size and overlap for document chunks, in tokens. Kept well under
    # EMBEDDING_MAX_TOKENS: retrieval quality degrades long before the encoder's
    # hard limit, because one vector has to represent the whole chunk.
    CHUNK_TARGET_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64

    @field_validator("LLM_PROVIDER", "EMBEDDING_PROVIDER", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        return v.lower().strip() if isinstance(v, str) else v


# ── Retrieval ───────────────────────────────────────────────
class RetrievalSettings(BaseSettings):
    # Candidates pulled from each arm BEFORE fusion. Deliberately wider than
    # the final context: retrieving exactly top_k makes a chunk ranked k+1
    # unrecoverable, which is what the previous limit=5 did.
    RETRIEVAL_CANDIDATES_PER_SOURCE: int = 25
    RETRIEVAL_MESSAGE_CANDIDATES: int = 5

    # Final context size handed to synthesis.
    RETRIEVAL_TOP_K: int = 5

    # Cosine-similarity floor for the dense arms. Applied per-source, where the
    # score is calibrated — an RRF score is not, so it cannot carry a floor.
    # Retrieving nothing is a valid, honest outcome; the previous code had no
    # floor and labelled whatever came back "relevant".
    #
    # THIS VALUE IS PROVIDER-SPECIFIC AND ONLY PARTLY CALIBRATED.
    # Measured on the committed golden set (scripts/run_retrieval_eval.py
    # --sweep-threshold) with all-MiniLM-L6-v2, k=3, 19 chunks:
    #
    #     floor   recall@3   abstains on unanswerable
    #     0.00-0.35  1.000        0 / 2
    #     0.40       0.969        1 / 2
    #     0.50       0.844        2 / 2
    #     0.60       0.562        2 / 2
    #
    # Two things follow. First, for MiniLM anything at or below 0.35 is inert —
    # 0.25 does nothing at all, and an unanswerable query ("configure SAML SSO",
    # absent from the corpus) still retrieved 3 chunks at 0.467. Second, NO
    # floor achieves both full recall and full abstention: topically-near but
    # unanswerable queries outscore some genuinely relevant chunks. A single
    # cosine threshold cannot separate them, which is the measured case for
    # Phase 3's cross-encoder reranker — it scores query-document relevance
    # directly rather than embedding proximity.
    #
    # The default below is UNCALIBRATED for the production provider
    # (text-embedding-3-small), whose similarity distribution differs from
    # MiniLM's. Re-run the sweep against it before trusting this number.
    RETRIEVAL_MIN_SIMILARITY: float = 0.25

    # Reciprocal Rank Fusion constant (Cormack et al. 2009).
    RRF_K: int = 60


# ── Sparse retrieval (Phase 3) ──────────────────────────────
class SparseRetrievalSettings(BaseSettings):
    # Dense-only retrieval fails on exact identifiers, error codes, and rare
    # proper nouns — precisely the "factual lookup" queries the classifier
    # routes to RAG. A release tag like "20260714-af" has no useful embedding
    # neighbourhood; it is a literal string match or nothing.
    SPARSE_ENABLED: bool = True
    SPARSE_CANDIDATES: int = 25

    # Postgres text-search configuration used by BOTH the query and the
    # generated `chunks.content_tsv` column. These MUST agree: a tsquery built
    # with 'simple' does not match a tsvector built with 'english', because
    # stemming happens at both ends. Changing this requires a migration that
    # re-generates content_tsv.
    SPARSE_TS_CONFIG: str = "english"

    # ts_rank_cd floor. Unlike cosine similarity, ts_rank_cd is unbounded and
    # length-normalized by the flag we pass; a value near zero means "the terms
    # occur but carry no weight". Kept low because the reranker, not this floor,
    # is what decides relevance in Phase 3.
    SPARSE_MIN_RANK: float = 0.0


# ── Reranking (Phase 3) ─────────────────────────────────────
class RerankSettings(BaseSettings):
    # A cross-encoder scores (query, document) jointly instead of comparing two
    # independently-produced vectors. Phase 5 measured why this is needed: no
    # single cosine floor both keeps recall and rejects topically-near but
    # unanswerable queries, because bi-encoder proximity is not relevance.
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # Fused candidates handed to the cross-encoder. Retrieve wide, rerank to
    # RETRIEVAL_TOP_K. Cost is linear in this number (~30 ms for 25 pairs on
    # CPU), so it is a latency dial, not a correctness one.
    RERANK_CANDIDATES: int = 25

    # Relevance floor on the cross-encoder score, applied AFTER reranking and
    # in place of trusting the cosine floor to abstain. bge-reranker emits a
    # raw logit; we apply a sigmoid so this is a probability in [0, 1].
    #
    # UNCALIBRATED until measured — see scripts/run_retrieval_eval.py
    # --sweep-rerank. 0.0 disables the floor entirely, which is the safe
    # default until the sweep has been run against the deployed model.
    RERANK_MIN_SCORE: float = 0.0

    # Reranking must never fail a request. When the model cannot load (no
    # weights cached, no network at boot, OOM), the pipeline keeps the fused
    # ordering and records the failure in the trace.
    RERANK_TIMEOUT_S: float = 10.0


# ── Context sufficiency / abstention (Phase 3→4) ────────────
class SufficiencySettings(BaseSettings):
    # Whether to answer at all. Measured twice in Phase 3 that this CANNOT be
    # done with a relevance score: neither RETRIEVAL_MIN_SIMILARITY nor
    # RERANK_MIN_SCORE separates answerable from unanswerable queries at any
    # threshold. Sufficiency is a different quantity from relevance
    # (arXiv 2411.06037) and needs its own signal.
    #
    # OFF by default until measured. Turning this on can only ADD refusals, and
    # a wrongly refused real question is worse than an answered bad one.
    SUFFICIENCY_ENABLED: bool = False

    # How many salient query terms may be absent from the retrieved context
    # before the context is judged insufficient. 0 is strict: every
    # content-bearing word in the question must appear somewhere in what was
    # retrieved. Raise it if the false-abstention rate on answerable queries is
    # unacceptable — that rate, not the abstention rate, is the binding metric.
    SUFFICIENCY_MAX_UNCOVERED_TERMS: int = 0


# ── Query rewriting (Phase 3) ───────────────────────────────
class QueryRewriteSettings(BaseSettings):
    # Deterministic, no-LLM expansion: acronym/identifier preservation and
    # stopword-trimming for the sparse arm. Free, so on by default.
    QUERY_EXPANSION_ENABLED: bool = True

    # HyDE and multi-query each cost an LLM call per request, which is exactly
    # the cost Phase 2 removed by deleting the ReAct loop. They are OFF by
    # default and must earn their place against the golden set before being
    # enabled — not turned on because the technique is well known.
    HYDE_ENABLED: bool = False
    MULTI_QUERY_ENABLED: bool = False
    MULTI_QUERY_COUNT: int = 3


class LiveKitSettings(BaseSettings):
    LIVEKIT_URL: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: SecretStr = SecretStr("")


class RAGServiceSettings(BaseSettings):
    RAG_SERVICE_TOKEN: SecretStr = SecretStr("")


# ── Operational (Phase 6) ───────────────────────────────────
class OpsSettings(BaseSettings):
    # Sliding-window limit per tenant per minute on the spend-bearing paths
    # (/rag/query, document ingest/replace). In-process — see
    # app/core/rate_limit.py for the single-process caveat and the Redis seam.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_RPM: int = 30

    # MemoryCrew substance gate. The crew costs 3-6 LLM calls per run and
    # previously fired on EVERY query — including "thanks". A query or answer
    # below these floors carries nothing worth remembering.
    MEMORY_MIN_QUERY_CHARS: int = 12
    MEMORY_MIN_ANSWER_CHARS: int = 40


# ── Synthesis (Phase 8) ─────────────────────────────────────
class SynthesisSettings(BaseSettings):
    # Which backend renders the final answer from retrieved context.
    #   "crewai" — the SupportCrew single tool-less agent (current behavior).
    #   "direct" — one chat-completion call through llm_provider.direct_chat.
    # DO NOT flip to "direct" without the Phase 9 RAGAS parity measurement;
    # the flip also changes `routed_to` deliberately (invariant 7).
    SYNTHESIS_BACKEND: str = "crewai"

    @field_validator("SYNTHESIS_BACKEND", mode="before")
    @classmethod
    def _normalize_backend(cls, v: str) -> str:
        return v.lower().strip() if isinstance(v, str) else v


# ── Semantic cache (Phase 7) ────────────────────────────────
class SemanticCacheSettings(BaseSettings):
    # Upper bound on how long a cached answer may be served. The primary
    # invalidation signals are the corpus epoch and the cache_version
    # fingerprint (app/services/semantic_cache.py); the TTL is the backstop
    # for staleness those cannot see — the world changing, not the corpus.
    SEMANTIC_CACHE_TTL_SECONDS: int = 7 * 24 * 3600

    # Vector-similarity floor for serving a cached answer to a NEW query.
    # None → a per-provider default in semantic_cache.py. This floor decides
    # whether two questions get the same answer, so it must be strict: on
    # short normalized queries encoders routinely exceed 0.95 for pairs
    # differing only by a negation or a single entity ("is X covered" vs
    # "is X NOT covered") — which is why the old hardcoded 0.95 was unsafe.
    # Set explicitly only after a calibration sweep against the deployed
    # embedding model.
    SEMANTIC_CACHE_SCORE_THRESHOLD: float | None = None


# ── Observability ───────────────────────────────────────────
class ObservabilitySettings(BaseSettings):
    LANGSMITH_API_KEY: SecretStr = SecretStr("")
    LANGSMITH_PROJECT: str = "contextflow_dev"


# NOTE: LoggerSettings (LOG_LEVEL, LOG_FORMAT_JSON) is removed — logging was
# never configured from it. Reintroduce alongside an actual dictConfig (Phase 6).


# ── Composed Settings ───────────────────────────────────────
class Settings(
    AppSettings,
    EnvironmentSettings,
    CryptSettings,
    PostgresSettings,
    Neo4jSettings,
    CORSSettings,
    AISettings,
    RetrievalSettings,
    SparseRetrievalSettings,
    RerankSettings,
    SufficiencySettings,
    QueryRewriteSettings,
    LiveKitSettings,
    RAGServiceSettings,
    OpsSettings,
    SynthesisSettings,
    SemanticCacheSettings,
    ObservabilitySettings,
):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Module-level singleton — import this everywhere
settings = Settings()
