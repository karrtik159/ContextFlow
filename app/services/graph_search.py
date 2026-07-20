"""
Neo4j graph database operations — entity and relationship queries.
"""

from __future__ import annotations

import threading

from neo4j import AsyncGraphDatabase

from app.core.config import settings

# ── Lazy, thread-safe Neo4j driver singleton ─────────────────
# The driver is created on first use instead of at import time,
# so importing this module no longer requires a running Neo4j.

_driver = None
_driver_lock = threading.Lock()


def _get_driver():
    """Return the shared async Neo4j driver, creating it on first call.

    Uses double-checked locking to avoid races when multiple threads
    (e.g. concurrent CrewAI tool calls) reach this simultaneously.
    """
    global _driver
    if _driver is not None:
        return _driver

    with _driver_lock:
        if _driver is not None:
            return _driver
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD.get_secret_value()),
        )
    return _driver


async def get_driver():
    """Return the shared async Neo4j driver (lazy-initialized)."""
    return _get_driver()


async def find_related_entities(
    entity_name: str,
    user_id: str,
    max_hops: int = 2,
) -> list[dict]:
    """
    Traverse the knowledge graph to find entities related to `entity_name`
    within `max_hops` relationship hops, confined to one user's subgraph.

    Args:
        entity_name: Name of the node to start traversal from.
        user_id: **Required** tenant scope. Both endpoints of every returned
            path must carry this ``user_id`` property.
        max_hops: Relationship hops to traverse, clamped to 1-3.

    Raises:
        ValueError: If ``user_id`` is falsy. The previous revision had no
            tenant predicate at all, so any caller traversed every user's
            nodes. Scoping is mandatory.

    Note:
        The ``user_id`` property is written by Mem0's Neo4j graph store, which
        owns ingestion into this graph. If Mem0 changes its scoping property,
        this query returns *nothing* rather than over-returning — it fails
        closed, which is the correct direction for a tenant boundary.
    """
    if not user_id:
        raise ValueError(
            "find_related_entities requires a user_id — an unscoped traversal "
            "would read across all tenants."
        )

    # Neo4j requires a literal int for variable-length path patterns, so
    # max_hops is interpolated. int() + clamp makes it non-injectable; keep
    # that cast if you touch this line. Every other value is parameterized.
    safe_hops = max(1, min(int(max_hops), 3))
    query = f"""
    MATCH path = (start {{name: $name, user_id: $user_id}})
                 -[*1..{safe_hops}]-
                 (end {{user_id: $user_id}})
    RETURN
        start.name   AS entity,
        [r IN relationships(path) | type(r)] AS relationships,
        end.name     AS related_entity,
        length(path) AS hops
    ORDER BY hops ASC
    LIMIT 20
    """
    driver = _get_driver()
    async with driver.session() as session:
        result = await session.run(query, name=entity_name, user_id=user_id)
        return [record.data() async for record in result]


async def close_driver():
    """Gracefully close the Neo4j driver (call on app shutdown)."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
