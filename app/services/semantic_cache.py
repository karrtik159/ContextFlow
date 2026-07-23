"""
User-scoped semantic response cache, backed by Neo4j.

Phase 7 hardening (docs/ADVANCED_RAG_PLAN.md, Phase 4 §cache + invariant 10):
an entry is servable only when THREE conditions hold, checked at lookup —

1. `corpus_epoch` matches the tenant's current epoch. Every corpus mutation
   bumps the epoch (app/services/corpus/events.py), so an answer grounded in
   deleted or replaced content stops being served the moment the corpus
   changes, even if the proactive delete below never ran.
2. `cache_version` matches the current model/prompt fingerprint. A provider,
   model, or synthesis-prompt change orphans every entry written before it.
3. `timestamp` is within SEMANTIC_CACHE_TTL_SECONDS. `c.timestamp` was
   previously written and never read; now it expires.

Entries that fail any condition are simply never returned; `corpus_changed`
also deletes the tenant's entries outright as hygiene. Pre-Phase-7 nodes have
none of these properties and therefore never match again — deliberate, since
nothing recorded what they were grounded in.
"""

import hashlib
import logging
import time

from app.services.graph_search import get_driver

logger = logging.getLogger(__name__)


def cache_version() -> str:
    """Fingerprint of everything that makes a cached answer reproducible.

    Embedding model (the vector space the similarity lookup runs in), LLM
    model, and the synthesis system prompt. Hashing the prompt TEXT rather
    than a hand-maintained version constant means a prompt edit cannot forget
    to invalidate.
    """
    from app.core.config import settings

    try:
        from app.services.rag_service import SYNTHESIS_SYSTEM_PROMPT as prompt
    except Exception:  # pragma: no cover - import cycle guard only
        prompt = "unversioned"
    basis = f"{settings.EMBEDDING_MODEL}|{settings.LLM_MODEL}|{prompt}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _min_valid_timestamp_ms() -> int:
    """Oldest servable entry, in ms since epoch — Neo4j `timestamp()` scale."""
    from app.core.config import settings

    return int(time.time() * 1000) - settings.SEMANTIC_CACHE_TTL_SECONDS * 1000


# Per-provider similarity floors, used when SEMANTIC_CACHE_SCORE_THRESHOLD is
# unset. Deliberately stricter than the old hardcoded 0.95: this floor decides
# whether two DIFFERENT questions receive the SAME cached answer, and near-
# duplicate queries differing by a negation or one entity routinely clear 0.95.
# MiniLM's similarity distribution runs hotter than OpenAI's on short text, so
# its floor is higher. CONSERVATIVE PENDING CALIBRATION — run a sweep against
# the deployed model before trusting these further than "safer than 0.95".
_PROVIDER_SCORE_THRESHOLDS = {
    "openai": 0.97,
    "huggingface": 0.985,
}
_FALLBACK_SCORE_THRESHOLD = 0.97


def _score_threshold() -> float:
    from app.core.config import settings

    configured = settings.SEMANTIC_CACHE_SCORE_THRESHOLD
    if configured is not None:
        return configured
    return _PROVIDER_SCORE_THRESHOLDS.get(
        settings.EMBEDDING_PROVIDER, _FALLBACK_SCORE_THRESHOLD
    )


async def init_semantic_cache():
    """
    Ensures that the necessary Vector Index exists in Neo4j for fast semantic lookups.
    Dynamically rebuilds the index if the underlying Embedder Dimension (e.g. OpenAI->HuggingFace) shifts.
    """
    from app.core.config import settings
    driver = await get_driver()
    target_dim = settings.EMBEDDING_DIMENSIONS

    async with driver.session() as session:
        # Check if index exists and what dimension it holds
        result = await session.run("SHOW VECTOR INDEXES YIELD name, options WHERE name = 'semantic_cache_index'")
        record = await result.single()
        should_create = True

        if record:
            try:
                options = record.get("options", {})
                index_config = options.get("indexConfig", {})
                current_dim = index_config.get("vector.dimensions")

                if current_dim and int(current_dim) != int(target_dim):
                    logger.warning(
                        "Vector dimension mismatch! Found %s, expected %s. Rebuilding index...",
                        current_dim,
                        target_dim,
                    )
                    await session.run("DROP INDEX semantic_cache_index")
                else:
                    should_create = False
            except Exception as e:
                logger.error(f"Error checking index dimensions: {e}")

        if should_create:
            query_create = f"""
            CREATE VECTOR INDEX semantic_cache_index IF NOT EXISTS
            FOR (c:SemanticCache) ON (c.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {target_dim},
              `vector.similarity_function`: 'cosine'
            }}}}
            """
            await session.run(query_create)
            logger.info("SemanticCache vector index initialized.")


async def get_cached_response(
    normalized_query: str,
    embedding: list[float],
    user_id: str,
    *,
    corpus_epoch: int = 0,
    candidate_count: int = 50,
    score_threshold: float | None = None,
) -> str | None:
    """
    Return a cached answer for this user only.

    Exact normalized-query matches win. If no exact match exists, use the vector
    index with a larger candidate set, then filter to the user's cache entries.

    `corpus_epoch` is the tenant's CURRENT epoch (from
    app.services.corpus.get_corpus_epoch_safe). Entries stamped with any other
    epoch — including pre-Phase-7 entries stamped with none — are not servable.
    """
    driver = await get_driver()

    # coalesce(-1) makes a node with no epoch property fail the comparison
    # against every real epoch (>= 0) instead of matching NULL semantics.
    validity = """
        AND coalesce(c.corpus_epoch, -1) = $corpus_epoch
        AND coalesce(c.cache_version, '') = $cache_version
        AND coalesce(c.timestamp, 0) >= $min_timestamp
    """

    exact_query = f"""
    MATCH (c:SemanticCache {{user_id: $user_id, normalized_query: $normalized_query}})
    WHERE true
    {validity}
    RETURN c.answer AS answer
    ORDER BY c.timestamp DESC
    LIMIT 1
    """

    vector_query = f"""
    CALL db.index.vector.queryNodes('semantic_cache_index', $candidate_count, $embedding)
    YIELD node AS c, score
    WHERE c.user_id = $user_id AND score > $score_threshold
    {validity}
    RETURN c.answer AS answer
    ORDER BY score DESC LIMIT 1
    """

    shared_params = {
        "user_id": user_id,
        "corpus_epoch": corpus_epoch,
        "cache_version": cache_version(),
        "min_timestamp": _min_valid_timestamp_ms(),
    }

    async with driver.session() as session:
        result = await session.run(
            exact_query,
            normalized_query=normalized_query,
            **shared_params,
        )
        record = await result.single()
        if record:
            return record["answer"]

        result = await session.run(
            vector_query,
            embedding=embedding,
            candidate_count=max(1, candidate_count),
            score_threshold=(
                score_threshold if score_threshold is not None else _score_threshold()
            ),
            **shared_params,
        )
        record = await result.single()
        if record:
            return record["answer"]
        return None


async def populate_semantic_cache(
    normalized_query: str,
    embedding: list[float],
    answer: str,
    user_id: str,
    session_id: str | None = None,
    *,
    corpus_epoch: int = 0,
):
    """
    Upsert a finalized RAG response in the user's semantic cache.

    `corpus_epoch` must be the epoch the answer was RETRIEVED at — the caller
    resolves it once per request — so that a corpus mutation between retrieval
    and this background task cannot stamp a stale answer as current.
    """
    driver = await get_driver()

    query = """
    MERGE (c:SemanticCache {
        user_id: $user_id,
        normalized_query: $normalized_query
    })
    SET
        c.embedding = $embedding,
        c.answer = $answer,
        c.session_id = $session_id,
        c.corpus_epoch = $corpus_epoch,
        c.cache_version = $cache_version,
        c.timestamp = timestamp()
    """

    params = {
        "normalized_query": normalized_query,
        "embedding": embedding,
        "answer": answer,
        "user_id": user_id,
        "session_id": session_id,
        "corpus_epoch": corpus_epoch,
        "cache_version": cache_version(),
    }

    async with driver.session() as session:
        await session.run(query, **params)


async def invalidate_user_cache(user_id: str) -> None:
    """Delete every cache entry this tenant owns.

    Hygiene, not correctness: called by `corpus_changed` after the epoch bump.
    An entry this delete misses (Neo4j down, race with a concurrent populate)
    is already unservable because its stored epoch no longer matches.
    """
    driver = await get_driver()

    query = """
    MATCH (c:SemanticCache {user_id: $user_id})
    DETACH DELETE c
    """

    async with driver.session() as session:
        await session.run(query, user_id=user_id)
    logger.info("Invalidated semantic cache for user=%s", user_id)
