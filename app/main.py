"""
FastAPI application entry point.

Uses `create_application` initialized with modular settings
and a fully configured lifespan from `app.core.setup.py`.
"""

import asyncio
import logging

from app.api.router import api_router
from app.core.config import settings
from app.core.setup import create_application

logger = logging.getLogger(__name__)

app = create_application(
    router=api_router,
    settings=settings,
    create_tables_on_start=True
)

_PROBE_TIMEOUT_S = 5.0


async def _probe_postgres() -> str | None:
    """None when healthy, an error summary when not."""
    from sqlalchemy import text

    from app.core.db import engine

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return None
    except Exception as exc:
        return f"{type(exc).__name__}"


async def _probe_neo4j() -> str | None:
    from app.services.graph_search import get_driver

    try:
        driver = await get_driver()
        await driver.verify_connectivity()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}"


async def _run_probe(probe) -> str | None:
    try:
        return await asyncio.wait_for(probe(), timeout=_PROBE_TIMEOUT_S)
    except TimeoutError:
        return "Timeout"


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness: the process is up. Says nothing about dependencies."""
    return {"status": "healthy", "environment": settings.ENVIRONMENT.value}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness: can this instance actually serve a knowledge query?

    Probes the stores the request path depends on (Phase 6 — the old /health
    returned "healthy" unconditionally, so an instance with no database
    passed every check while failing every request). Error DETAIL stays in
    the logs; the response carries only the exception class name.
    """
    from fastapi.responses import JSONResponse

    postgres_error, neo4j_error = await asyncio.gather(
        _run_probe(_probe_postgres), _run_probe(_probe_neo4j)
    )
    dependencies = {
        "postgres": postgres_error or "ok",
        "neo4j": neo4j_error or "ok",
    }
    ready = postgres_error is None and neo4j_error is None
    if not ready:
        logger.warning("Readiness probe failing: %s", dependencies)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "dependencies": dependencies},
    )


@app.get("/health/metrics", tags=["Health"])
async def metrics_snapshot():
    """Per-process operational counters (see app/core/metrics.py for scope)."""
    from app.core.metrics import snapshot

    return snapshot()
