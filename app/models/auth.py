from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    org_id: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str
    org_id: Optional[str] = None
    org_code: Optional[str] = None

class RegisterRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str

class TokenDocument(BaseModel):
    user_id: str
    access_token: str
    refresh_token: str
    is_revoked: bool = False
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
