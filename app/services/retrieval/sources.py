"""
Retrieval arms. Each returns a ranked, tenant-scoped list of RetrievedChunk.

Every function here takes `user_id` as a REQUIRED argument and raises without
it. That is not defensive style, it is the fix for a shipped bug: a conditional
tenant filter silently degrades into a cross-tenant scan, which is exactly how
the original defect reached production (docs/ADVANCED_RAG_PLAN.md §5.2).

No function here is reachable by an LLM. There is no tool wrapper, no
args_schema, and no prompt-supplied argument — the pipeline calls these directly
with a scope resolved from the request. This is strictly stronger than the tool
design it replaces, where scope was at least a pydantic field the model could
not address; here the model is not in the call path at all.
"""

from __future__ import annotations

import logging
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession
from app.models.document import Chunk
from app.models.message import Message
from app.services.retrieval.contracts import RetrievedChunk

logger = logging.getLogger(__name__)

# pgvector's HNSW scan walks `hnsw.ef_search` candidates and applies the WHERE
# clause afterwards, so the default of 40 can return nothing for a tenant whose
# rows are not in the global top-40. Raising it per-session trades a little
# latency for recall that would otherwise silently vanish as the table grows.
HNSW_EF_SEARCH = 200


def _require_user_id(user_id: uuid.UUID | None, fn: str) -> uuid.UUID:
    if not user_id:
        raise ValueError(
            f"{fn} requires a user_id — an unscoped search would read across all tenants."
        )
    return user_id


async def _set_ef_search(db: AsyncSession) -> None:
    from sqlalchemy import text as sqltext

    try:
        await db.execute(sqltext(f"SET LOCAL hnsw.ef_search = {int(HNSW_EF_SEARCH)}"))
    except Exception as exc:  # pragma: no cover - depends on pgvector version
        # Not fatal: an older pgvector without this GUC still returns results,
        # just with default recall. Log rather than fail the whole retrieval.
        logger.debug("Could not set hnsw.ef_search: %s", exc)


async def search_chunks(
    db: AsyncSession,
    *,
    query_embedding: list[float],
    user_id: uuid.UUID,
    limit: int = 25,
    min_similarity: float = 0.0,
) -> list[RetrievedChunk]:
    """Dense retrieval over the document corpus.

    This is the primary arm: it searches `chunks`, which is the actual corpus,
    rather than whole chat messages.

    `min_similarity` is a relevance floor applied here rather than after fusion,
    because cosine similarity is calibrated and an RRF score is not. Returning
    nothing is a valid answer.
    """
    _require_user_id(user_id, "search_chunks")
    await _set_ef_search(db)

    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk, distance.label("distance"))
        # The tenant filter runs against chunks.user_id directly — never via a
        # join to documents. See the comment on Chunk.user_id for why.
        .where(Chunk.user_id == user_id, Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )

    rows = (await db.execute(stmt)).all()

    results: list[RetrievedChunk] = []
    for chunk, dist in rows:
        similarity = 1.0 - float(dist)
        if similarity < min_similarity:
            continue
        results.append(
            RetrievedChunk(
                text=chunk.text,
                source="vector",
                rank=len(results) + 1,
                score=similarity,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                heading_path=chunk.heading_path,
                metadata={"chunk_index": chunk.chunk_index},
            )
        )
    return results


async def search_messages(
    db: AsyncSession,
    *,
    query_embedding: list[float],
    user_id: uuid.UUID,
    limit: int = 5,
    min_similarity: float = 0.0,
) -> list[RetrievedChunk]:
    """Dense retrieval over the user's own chat history.

    A SECONDARY personalization signal, not the corpus. Weighted below the
    corpus arm during fusion: past conversation says what this user cares about,
    not what is true.

    NOTE: the tenant filter here goes through a join to chat_sessions, because
    messages carry no denormalized user_id. That is the same HNSW pre-filter
    weakness documented on Chunk.user_id and it applies to this query — recall
    can degrade as the message table grows. Fixing it means a migration on
    `messages`; it is deliberately out of Phase 2's scope and this arm is a
    supplementary signal, so degraded recall here is not a correctness problem.
    """
    _require_user_id(user_id, "search_messages")
    await _set_ef_search(db)

    distance = Message.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Message, distance.label("distance"))
        .join(Message.session)
        .where(ChatSession.user_id == user_id, Message.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )

    rows = (await db.execute(stmt)).all()

    results: list[RetrievedChunk] = []
    for message, dist in rows:
        similarity = 1.0 - float(dist)
        if similarity < min_similarity:
            continue
        results.append(
            RetrievedChunk(
                text=message.content,
                source="messages",
                rank=len(results) + 1,
                score=similarity,
                heading_path=f"Past conversation ({message.role})",
                metadata={"role": message.role},
            )
        )
    return results


# Crude entity candidates: capitalised words and quoted phrases. The previous
# design had an LLM choose what to look up, which cost a model turn per query.
# This is deliberately dumb and deterministic; Phase 3's query rewriting is
# where entity selection gets better.
_CAPITALISED = re.compile(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b")
_QUOTED = re.compile(r"[\"'`]([^\"'`]{3,64})[\"'`]")
_STOPWORDS = {"The", "This", "That", "What", "When", "Where", "Why", "How", "Who", "Is", "Are", "Does", "Do", "Can", "Should", "Could", "Would", "If", "In", "On", "At", "For", "And", "But", "Or"}


def extract_entity_candidates(query: str, *, limit: int = 3) -> list[str]:
    """Pick likely entity names out of a query, without an LLM call."""
    candidates: list[str] = []
    seen: set[str] = set()

    for match in _QUOTED.finditer(query):
        value = match.group(1).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            candidates.append(value)

    for match in _CAPITALISED.finditer(query):
        value = match.group(1)
        if value in _STOPWORDS or value.lower() in seen:
            continue
        seen.add(value.lower())
        candidates.append(value)

    return candidates[:limit]


async def search_graph(
    *,
    query: str,
    user_id: str,
    max_hops: int = 2,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """Knowledge-graph arm — entities related to those named in the query.

    Returns an empty list rather than raising when Neo4j is unreachable or the
    query names no entities: one arm failing must degrade the fused result, not
    fail the request.
    """
    if not user_id:
        raise ValueError("search_graph requires a user_id — an unscoped traversal reads all tenants.")

    entities = extract_entity_candidates(query)
    if not entities:
        return []

    from app.services.graph_search import find_related_entities

    results: list[RetrievedChunk] = []
    for entity in entities:
        try:
            records = await find_related_entities(entity, user_id=user_id, max_hops=max_hops)
        except Exception as exc:
            logger.warning("Graph search failed for entity %r: %s", entity, exc)
            continue

        for record in records:
            rels = " -> ".join(record.get("relationships") or [])
            text = (
                f"{record.get('entity')} {rels} {record.get('related_entity')}".strip()
            )
            if not text:
                continue
            results.append(
                RetrievedChunk(
                    text=text,
                    source="graph",
                    rank=len(results) + 1,
                    # Closer hops rank higher; 1/hops keeps the raw score
                    # monotonic with usefulness even though it is not
                    # comparable to a cosine similarity.
                    score=1.0 / max(1, int(record.get("hops") or 1)),
                    heading_path="Knowledge graph",
                    metadata={"hops": record.get("hops"), "seed_entity": entity},
                )
            )
            if len(results) >= limit:
                return results
    return results


async def search_memory(*, query: str, user_id: str, limit: int = 5) -> list[RetrievedChunk]:
    """Mem0 arm — this user's remembered preferences and history.

    The Mem0 SDK is blocking, so it is wrapped in a thread (CLAUDE.md: async
    boundaries). Failures degrade to an empty arm.
    """
    if not user_id:
        raise ValueError("search_memory requires a user_id — memories are per-user.")

    import asyncio

    from app.memory.mem0_service import Mem0Service

    try:
        raw = await asyncio.to_thread(
            Mem0Service.search_memories, query=query, user_id=user_id, limit=limit
        )
    except Exception as exc:
        logger.warning("Memory search failed for user=%s: %s", user_id, exc)
        return []

    # Mem0 returns either a bare list or {"results": [...]} depending on
    # version; normalise rather than assume.
    items = raw.get("results", []) if isinstance(raw, dict) else (raw or [])

    results: list[RetrievedChunk] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("memory") or item.get("text") or ""
            score = float(item.get("score") or 0.0)
        else:
            text, score = str(item), 0.0
        if not text.strip():
            continue
        results.append(
            RetrievedChunk(
                text=text.strip(),
                source="memory",
                rank=len(results) + 1,
                score=score,
                heading_path="User memory",
            )
        )
    return results
