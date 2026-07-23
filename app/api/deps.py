"""
Shared FastAPI dependencies — DB session, current user, rate limiting.

All dependencies are designed as reusable `Annotated` type aliases:
    DBSession       → async database session (auto commit/rollback)
    CurrentUser     → authenticated user dict (requires valid JWT)
    OptionalUser    → user dict or None (no auth required)
"""

import hmac
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db
from app.services.crud import user_crud

# ── OAuth2 scheme — extracts Bearer token from Authorization header ──
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login")


# ── Database Session ─────────────────────────────────────────────────
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ── Current Authenticated User ───────────────────────────────────────
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: DBSession,
) -> dict[str, Any]:
    """
    Verify JWT (access type, not blacklisted), look up the user,
    and return their data as a dict.
    Raises 401 if the token is invalid, blacklisted, or the user doesn't exist.
    """
    from app.core.security import TokenType, verify_token

    token_data = await verify_token(token, TokenType.ACCESS, db)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # JWT sub claim stores the username
    user = await user_crud.get(db, username=token_data.username_or_email)
    if not user:
        # Fallback: try by email (in case sub was set to email)
        user = await user_crud.get(db, email=token_data.username_or_email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


# ── Optional User (no auth required) ────────────────────────────────
async def get_optional_user(
    request: Request,
    db: DBSession,
) -> dict[str, Any] | None:
    """
    ABSENT credentials mean anonymous; PRESENTED-but-invalid credentials are
    an error (Phase 6). The old behavior — swallowing the 401 and degrading to
    anonymous — turned an expired token into a silent loss of scope: the
    caller kept getting 200s while quietly reading an empty corpus, and a
    stolen-but-revoked token probed endpoints indistinguishably from a guest.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # get_current_user raises 401 for invalid/expired/revoked — let it.
    return await get_current_user(parts[1], db)


OptionalUser = Annotated[dict[str, Any] | None, Depends(get_optional_user)]


def is_valid_rag_service_request(request: Request) -> bool:
    """Return True when an internal worker presents the configured RAG service token."""
    expected = settings.RAG_SERVICE_TOKEN.get_secret_value()
    supplied = request.headers.get("X-RAG-Service-Token", "")
    return bool(expected) and hmac.compare_digest(supplied, expected)


def get_user_id_value(user: Any) -> str | None:
    """Extract a stable user id string from the auth dependency result."""
    if user is None:
        return None
    if isinstance(user, dict):
        value = user.get("id")
    else:
        value = getattr(user, "id", None)
    return str(value) if value else None


def resolve_rag_user_id(
    *,
    request_user_id: str | None,
    user: Any,
    is_service_request: bool,
) -> str | None:
    """Resolve the authoritative user scope for RAG/context endpoints.

    User JWT identity wins for app clients. Internal service-token callers may
    supply a user id explicitly. Anonymous callers cannot opt into a user scope.
    """
    auth_user_id = get_user_id_value(user)
    supplied_user_id = None if request_user_id in (None, "", "anonymous") else str(request_user_id)

    if is_service_request:
        return supplied_user_id or auth_user_id

    if auth_user_id:
        if supplied_user_id and supplied_user_id != auth_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="request user_id does not match authenticated user",
            )
        return auth_user_id

    if supplied_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user_id requires authentication or internal service token",
        )
    return None
