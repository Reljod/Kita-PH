from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime, timezone

class UserBase(BaseModel):
    first_name: str
    last_name: str

class UserCreate(UserBase):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class PasswordUpdate(BaseModel):
    old_password: str
    new_password: str

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
