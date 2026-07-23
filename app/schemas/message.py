"""
Pydantic V2 schemas for Message — Create / Read.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageCreate(BaseModel):
    """API-facing schema — used for request validation."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class MessageCreateInternal(MessageCreate):
    """Internal schema — includes server-resolved fields for FastCRUD create().

    `user_id` is the session owner, resolved server-side (Phase 9) — never part
    of the API-facing `MessageCreate`, so a client cannot set it.
    """

    session_id: uuid.UUID
    user_id: uuid.UUID
    embedding: list[float] | None = None
    embedding_truncated: bool = False


class MessageRead(BaseModel):
    """Response schema for a single message."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    token_count: int | None
    created_at: datetime
