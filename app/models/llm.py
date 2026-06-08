from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone

class LlmCreateRequest(BaseModel):
    name: str
    model: str
    provider: str = 'openrouter'

class LlmResponse(BaseModel):
    id: str
    name: str
    model: str
    provider: str
    created_at: datetime
    updated_at: datetime

class LlmDocument(BaseModel):
    org_id: Optional[str] = None
    name: str
    model: str
    provider: str = 'openrouter'
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
