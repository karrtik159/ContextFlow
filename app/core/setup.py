import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import _AsyncGeneratorContextManager, asynccontextmanager
from typing import Any

import fastapi
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app.core.config import EnvironmentOption, Settings
from app.core.db import Base, engine
from app.core.request_context import RequestIDMiddleware, install_record_factory
from app.core.telemetry import init_telemetry, shutdown_telemetry
from app.services.embeddings import init_local_embedding_model
from app.services.graph_search import close_driver
from app.services.semantic_cache import init_semantic_cache

logger = logging.getLogger(__name__)

# The shipped default from config.py. Compared verbatim at boot: a deployment
# still carrying it mints forgeable tokens.
_DEFAULT_SECRET_KEY = "super-secret-change-me-in-production"


def enforce_boot_security(settings: Settings) -> None:
    """Refuse to boot a non-local deployment with the default SECRET_KEY.

    Every JWT this app mints is signed with it; a known key means anyone can
    forge any user. Local development warns instead of failing, because a
    fresh clone must still start.
    """
    if settings.SECRET_KEY.get_secret_value() != _DEFAULT_SECRET_KEY:
        return
    if settings.ENVIRONMENT == EnvironmentOption.LOCAL:
        logger.warning(
            "SECRET_KEY is the shipped default — fine for local, forgeable "
            "anywhere else. Set a real key before deploying."
        )
        return
    raise RuntimeError(
        f"Refusing to start in {settings.ENVIRONMENT.value!r} with the default "
        "SECRET_KEY: every token it signs is forgeable. Set SECRET_KEY."
    )


def resolve_cors_credentials(settings: Settings) -> bool:
    """Credentialed CORS and a wildcard origin must never combine.

    Starlette 'helpfully' echoes the request Origin when allow_origins=["*"]
    is paired with allow_credentials=True, which defeats the browser's
    protection entirely — any site can make credentialed calls. If the origin
    list is a wildcard, credentials are disabled and that choice is logged.
    """
    if "*" in settings.CORS_ORIGINS:
        logger.warning(
            "CORS_ORIGINS contains '*'; disabling allow_credentials so the "
            "wildcard cannot be combined with cookies/Authorization. List "
            "explicit origins to re-enable credentialed CORS."
        )
        return False
    return True


# -------------- database --------------
async def create_tables() -> None:
    """Create database tables if they do not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# -------------- factory --------------
def lifespan_factory(
    settings: Settings,
    create_tables_on_start: bool = True,
) -> Callable[[FastAPI], _AsyncGeneratorContextManager[Any]]:
    """Factory to create a lifespan async context manager for a FastAPI app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        logger.info("Starting %s (%s)", settings.APP_NAME, settings.ENVIRONMENT.value)

        try:
            init_telemetry()

            if create_tables_on_start:
                await create_tables()

            await init_semantic_cache()

            init_local_embedding_model()

            yield

        finally:
            shutdown_telemetry()

            logger.info("Shutting down...")
            await close_driver()

    return lifespan


# -------------- application --------------
def create_application(
    router: APIRouter,
    settings: Settings,
    create_tables_on_start: bool = True,
    lifespan: Callable[[FastAPI], _AsyncGeneratorContextManager[Any]] | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    kwargs.update(
        {
            "title": settings.APP_NAME,
            "description": settings.APP_DESCRIPTION,
            "version": settings.APP_VERSION,
        }
    )
    if settings.CONTACT_NAME or settings.CONTACT_EMAIL:
        kwargs.update({"contact": {"name": settings.CONTACT_NAME, "email": settings.CONTACT_EMAIL}})
    if settings.LICENSE_NAME:
        kwargs.update({"license_info": {"name": settings.LICENSE_NAME}})

    # Default docs are disabled here and re-registered below for non-production
    # environments, so that production serves no docs routes at all.
    kwargs.update({"docs_url": None, "redoc_url": None, "openapi_url": None})

    if lifespan is None:
        lifespan = lifespan_factory(settings, create_tables_on_start=create_tables_on_start)

    enforce_boot_security(settings)
    install_record_factory()

    application = FastAPI(lifespan=lifespan, **kwargs)
    application.include_router(router)

    application.add_middleware(RequestIDMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=resolve_cors_credentials(settings),
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )

    if settings.ENVIRONMENT != EnvironmentOption.PRODUCTION:
        docs_router = APIRouter()

        @docs_router.get("/docs", include_in_schema=False)
        async def get_swagger_documentation() -> fastapi.responses.HTMLResponse:
            return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

        @docs_router.get("/redoc", include_in_schema=False)
        async def get_redoc_documentation() -> fastapi.responses.HTMLResponse:
            return get_redoc_html(openapi_url="/openapi.json", title="docs")

        @docs_router.get("/openapi.json", include_in_schema=False)
        async def openapi() -> dict[str, Any]:
            return get_openapi(
                title=application.title,
                version=application.version,
                routes=application.routes,
            )

        application.include_router(docs_router)

    return application
