from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime, timezone

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    org_id: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    org_id: Optional[str] = Field(None, max_length=100)
    org_code: Optional[str] = Field(None, max_length=100)

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)

class TokenDocument(BaseModel):
    user_id: str
    access_token: str
    refresh_token: str
    is_revoked: bool = False
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
