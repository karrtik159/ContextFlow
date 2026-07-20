"""
User CRUD endpoints.

Adapted to match benavlabs/FastAPI-boilerplate for standard User ops.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DBSession
from app.core.security import hash_password
from app.schemas.user import UserCreate, UserCreateInternal, UserRead, UserUpdate
from app.services.crud import user_crud

router = APIRouter(tags=["users"])


@router.post("/user", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreate, db: DBSession):
    """Register a new user."""
    existing = await user_crud.get(db, email=payload.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    internal = UserCreateInternal(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
    )
    new_user = await user_crud.create(
        db,
        object=internal,
        schema_to_select=UserRead,
        return_as_model=True,
    )
    return new_user


@router.get("/user/me", response_model=UserRead)
async def get_users_me(current_user: CurrentUser):
    """Return the currently authenticated user's profile."""
    return UserRead.model_validate(current_user)


@router.get("/user/{user_id}", response_model=UserRead)
async def get_user(user_id: UUID, db: DBSession):
    """Retrieve a single user by ID."""
    user = await user_crud.get(
        db,
        schema_to_select=UserRead,
        return_as_model=True,
        id=user_id,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/user/{user_id}", response_model=UserRead)
async def update_user(user_id: UUID, payload: UserUpdate, db: DBSession):
    """Partially update a user, safely skipping internal fields on the schema."""
    update_data = payload.model_dump(exclude_unset=True)
    
    # Map plain-text password to hashed_password for the ORM
    if "password" in update_data:
        update_data["hashed_password"] = hash_password(update_data.pop("password"))

    # Protect against raw internal fields if accidentally merged in payload
    if "id" in update_data:
        update_data.pop("id")
        
    updated = await user_crud.update(
        db,
        object=update_data,
        schema_to_select=UserRead,
        return_as_model=True,
        id=user_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated
