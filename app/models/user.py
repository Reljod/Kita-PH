from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime, timezone

class UserBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)

class UserCreate(UserBase):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)

class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)

class PasswordUpdate(BaseModel):
    old_password: str = Field(..., min_length=6, max_length=100)
    new_password: str = Field(..., min_length=6, max_length=100)

class UserResponse(UserBase):
    id: str
    email: EmailStr
    created_at: datetime
    updated_at: datetime

class UserDocument(UserBase):
    email: EmailStr
    password: str  # Hashed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
