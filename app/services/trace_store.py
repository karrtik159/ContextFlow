"""
Trace persistence.

Runs as a BackgroundTask, so it owns its own session: the request's session is
closed by the time this executes. Failures are logged and swallowed — losing an
observability record must never turn a served answer into an error.
"""

from __future__ import annotations

import logging

from app.services.retrieval.contracts import RetrievalTrace

logger = logging.getLogger(__name__)


async def persist_trace(
    *,
    trace: RetrievalTrace,
    routed_to: str,
    answer_hash: str | None = None,
) -> None:
    """Write one retrieval trace. Best-effort by design."""
    from app.core.db import async_session
    from app.models.retrieval_trace import RetrievalTraceRecord

    try:
        async with async_session() as db:
            db.add(
                RetrievalTraceRecord(
                    id=trace.trace_id,
                    user_id=str(trace.user_id),
                    original_query=trace.original_query,
                    normalized_query=trace.normalized_query,
                    rewritten_query=trace.rewritten_query,
                    routed_to=routed_to,
                    total_latency_ms=trace.total_latency_ms,
                    stages=trace.stages_as_json(),
                    final_chunk_ids=[
                        c.chunk_id for c in trace.final_chunks if c.chunk_id is not None
                    ],
                    answer_hash=answer_hash,
                )
            )
            await db.commit()
    except Exception:
        logger.exception("Failed to persist retrieval trace %s", trace.trace_id)
