"""
FastAPI application entry point.

Uses `create_application` initialized with modular settings
and a fully configured lifespan from `app.core.setup.py`.
"""

from app.api.router import api_router
from app.core.config import settings
from app.core.setup import create_application

app = create_application(
    router=api_router,
    settings=settings,
    create_tables_on_start=True
)

@app.get("/health", tags=["Health"])
async def health_check():
    """Application health endpoint."""
    return {"status": "healthy", "environment": settings.ENVIRONMENT.value}
