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
    # Below this, a result is noise and is dropped even if nothing replaces it.
    # Retrieving nothing is a valid, honest outcome; the previous code had no
    # floor and labelled whatever came back "relevant".
    RETRIEVAL_MIN_SIMILARITY: float = 0.25

    # Reciprocal Rank Fusion constant (Cormack et al. 2009).
    RRF_K: int = 60


class LiveKitSettings(BaseSettings):
    LIVEKIT_URL: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: SecretStr = SecretStr("")


class RAGServiceSettings(BaseSettings):
    RAG_SERVICE_TOKEN: SecretStr = SecretStr("")


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
    LiveKitSettings,
    RAGServiceSettings,
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
