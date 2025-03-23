from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from ..database.connection import get_db
from ..database.repositories import UserRepository

class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """User update model."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)

class UserProfile(UserBase):
    """User profile model."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class User(UserBase):
    """User model with password."""
    id: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

    @classmethod
    async def get_by_username(cls, username: str) -> Optional["User"]:
        """Get user by username."""
        db = get_db()
        async with db.get_session() as session:
            repo = UserRepository(session)
            return await repo.get_by_username(username)

    @classmethod
    async def get_by_email(cls, email: str) -> Optional["User"]:
        """Get user by email."""
        db = get_db()
        async with db.get_session() as session:
            repo = UserRepository(session)
            return await repo.get_by_email(email)

    @classmethod
    async def create(cls, **kwargs) -> "User":
        """Create new user."""
        db = get_db()
        async with db.get_session() as session:
            repo = UserRepository(session)
            return await repo.create(kwargs)

    async def update(self, **kwargs) -> "User":
        """Update user."""
        db = get_db()
        async with db.get_session() as session:
            repo = UserRepository(session)
            return await repo.update(self.id, kwargs)

    @classmethod
    async def delete(cls, user_id: str) -> bool:
        """Delete user."""
        db = get_db()
        async with db.get_session() as session:
            repo = UserRepository(session)
            return await repo.delete(user_id) 