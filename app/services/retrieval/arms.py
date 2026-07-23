"""
Arm registry — one declaration per retrieval source.

Before this, adding an arm meant three synchronized edits: the fan-out list in
`run_retrieval`, the `SOURCE_WEIGHTS` dict, and the `SourceName` literal. The
registry collapses the first two into one `ArmSpec`; the literal remains the
type-level contract in `contracts.py`.

Two deliberate indirections:

- `fn_name`, not a function reference. The pipeline resolves the name on ITS
  OWN module namespace at call time (`globals()[spec.fn_name]`), because
  `pipeline.search_chunks` is the established patch seam — the stage tests
  stub every arm by setting attributes on the pipeline module, and a registry
  holding direct references would silently bypass those stubs.

- `build(ctx, fn)` returns a factory taking the DB session (ignored by
  external arms). Settings are read inside the built coroutine, not at import,
  so per-test settings monkeypatches behave exactly as they did when the
  fan-out was inline.

The registry declares; the pipeline still decides HOW arms run (db arms
serially on the shared session inside savepoints, external arms concurrently)
based on `kind` — that execution split is a property of the session model, not
of any single arm.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.retrieval.contracts import RetrievedChunk

ArmFactory = Callable[[AsyncSession | None], Awaitable[list[RetrievedChunk]]]


@dataclass(frozen=True)
class ArmContext:
    """Everything a build function may close over for one retrieval run."""

    user_id: str
    scoped_uuid: uuid.UUID
    retrieval_query: str
    sparse_query: str
    query_embedding: list[float]


@dataclass(frozen=True)
class ArmSpec:
    """One retrieval source, declaratively.

    `name` is load-bearing three ways: the `source` stamped on every returned
    chunk, the `retrieve:<name>` trace stage label, and the RRF weight key.
    """

    name: str
    kind: Literal["db", "external"]
    weight: float
    fn_name: str
    enabled: Callable[[], bool]
    build: Callable[[ArmContext, Callable], ArmFactory]


def _always() -> bool:
    return True


def _build_vector(ctx: ArmContext, fn: Callable) -> ArmFactory:
    async def run(db):
        return await fn(
            db,
            query_embedding=ctx.query_embedding,
            user_id=ctx.scoped_uuid,
            limit=settings.RETRIEVAL_CANDIDATES_PER_SOURCE,
            min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
        )

    return run


def _build_messages(ctx: ArmContext, fn: Callable) -> ArmFactory:
    async def run(db):
        return await fn(
            db,
            query_embedding=ctx.query_embedding,
            user_id=ctx.scoped_uuid,
            limit=settings.RETRIEVAL_MESSAGE_CANDIDATES,
            min_similarity=settings.RETRIEVAL_MIN_SIMILARITY,
        )

    return run


def _build_sparse(ctx: ArmContext, fn: Callable) -> ArmFactory:
    async def run(db):
        return await fn(
            db,
            query=ctx.sparse_query,
            user_id=ctx.scoped_uuid,
            limit=settings.SPARSE_CANDIDATES,
            min_rank=settings.SPARSE_MIN_RANK,
            ts_config=settings.SPARSE_TS_CONFIG,
        )

    return run


def _build_graph(ctx: ArmContext, fn: Callable) -> ArmFactory:
    async def run(_db):
        return await fn(query=ctx.retrieval_query, user_id=ctx.user_id)

    return run


def _build_memory(ctx: ArmContext, fn: Callable) -> ArmFactory:
    async def run(_db):
        return await fn(query=ctx.retrieval_query, user_id=ctx.user_id)

    return run


# Order matters: it is the trace's stage order (db arms first, then external),
# preserved exactly from the pre-registry fan-out.
#
# Weights: the corpus is the authority; conversation history and memory
# personalise but should not outvote it. Graph sits between: structural, but
# sparse and noisy.
ARM_REGISTRY: tuple[ArmSpec, ...] = (
    ArmSpec("vector", "db", 1.0, "search_chunks", _always, _build_vector),
    ArmSpec("messages", "db", 0.4, "search_messages", _always, _build_messages),
    ArmSpec("bm25", "db", 1.0, "search_chunks_sparse", lambda: settings.SPARSE_ENABLED, _build_sparse),
    ArmSpec("graph", "external", 0.6, "search_graph", _always, _build_graph),
    ArmSpec("memory", "external", 0.5, "search_memory", _always, _build_memory),
)


def source_weights() -> dict[str, float]:
    """RRF weights, derived — a spec cannot exist without a weight."""
    return {spec.name: spec.weight for spec in ARM_REGISTRY}
