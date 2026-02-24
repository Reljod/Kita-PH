from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

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
    name: str
    model: str
    provider: str = 'openrouter'
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
