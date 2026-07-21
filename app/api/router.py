"""
Aggregates all v1 API routers into a single router.
"""

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.context import router as context_router
from app.api.v1.documents import router as documents_router
from app.api.v1.livekit_rooms import router as livekit_router
from app.api.v1.login import router as login_router
from app.api.v1.logout import router as logout_router
from app.api.v1.memory import router as memory_router
from app.api.v1.rag import router as rag_router
from app.api.v1.users import router as users_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(login_router)
api_router.include_router(logout_router)
api_router.include_router(users_router)
api_router.include_router(chat_router)
api_router.include_router(livekit_router)
api_router.include_router(rag_router)
api_router.include_router(memory_router)
api_router.include_router(context_router)
api_router.include_router(documents_router)
