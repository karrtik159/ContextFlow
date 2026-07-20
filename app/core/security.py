"""
JWT token creation, verification, password hashing, and authentication.

Features:
  - bcrypt password hashing with SHA-256 pre-hash (sidesteps 72-byte limit)
  - Access + Refresh token creation with embedded token type
  - Token verification with blacklist checking
  - authenticate_user() supports login by email OR username
  - Token blacklisting for logout
"""

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Literal

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.schemas.token import TokenBlacklistCreate, TokenData
from app.services.crud import token_blacklist_crud, user_crud


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


# ── Password Hashing ────────────────────────────────────────


def _prehash(password: str) -> bytes:
    """Pre-hash password with SHA-256 before bcrypt.

    Returns base64-encoded bytes (always 44 chars, under bcrypt's 72-byte limit).
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt with SHA-256 pre-hashing."""
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    return bcrypt.checkpw(_prehash(plain), hashed.encode("ascii"))


# ── Authentication ───────────────────────────────────────────


async def authenticate_user(
    username_or_email: str,
    password: str,
    db: AsyncSession,
) -> dict[str, Any] | Literal[False]:
    """Authenticate by email or username. Returns user dict or False."""
    if "@" in username_or_email:
        db_user = await user_crud.get(db=db, email=username_or_email)
    else:
        db_user = await user_crud.get(db=db, username=username_or_email)

    if not db_user:
        return False

    if not verify_password(password, db_user["hashed_password"]):
        return False

    return db_user


# ── Token Creation ───────────────────────────────────────────


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "token_type": TokenType.ACCESS})
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY.get_secret_value(),
        algorithm=settings.ALGORITHM,
    )


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire, "token_type": TokenType.REFRESH})
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY.get_secret_value(),
        algorithm=settings.ALGORITHM,
    )


# ── Token Verification ──────────────────────────────────────


def _decode_jwt(token: str) -> dict | None:
    """Shared JWT decode — returns payload dict or None on any error."""
    try:
        return jwt.decode(
            token,
            settings.SECRET_KEY.get_secret_value(),
            algorithms=[settings.ALGORITHM],
        )
    except JWTError:
        return None


async def verify_token(
    token: str,
    expected_token_type: TokenType,
    db: AsyncSession,
) -> TokenData | None:
    """Verify a JWT token: decode, check blacklist, validate type.

    Returns TokenData if valid, None otherwise.
    """
    # 1. Check blacklist
    is_blacklisted = await token_blacklist_crud.exists(db, token=token)
    if is_blacklisted:
        return None

    # 2. Decode
    payload = _decode_jwt(token)
    if payload is None:
        return None

    username_or_email: str | None = payload.get("sub")
    token_type: str | None = payload.get("token_type")

    if username_or_email is None or token_type != expected_token_type:
        return None

    return TokenData(username_or_email=username_or_email)


# ── Token Blacklisting ──────────────────────────────────────


async def blacklist_token(token: str, db: AsyncSession) -> None:
    """Add a single token to the blacklist."""
    payload = _decode_jwt(token)
    if payload is None:
        return  # Token already invalid, no need to blacklist

    exp_timestamp = payload.get("exp")
    if exp_timestamp is not None:
        expires_at = datetime.fromtimestamp(exp_timestamp, tz=UTC)
        await token_blacklist_crud.create(
            db,
            object=TokenBlacklistCreate(token=token, expires_at=expires_at),
        )
