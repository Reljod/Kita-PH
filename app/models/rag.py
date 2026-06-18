from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime, timezone

class RagCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=50000)
    agent_id: Optional[str] = Field(None, max_length=100)

class RagUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1, max_length=50000)

class RagResponse(BaseModel):
    id: str
    title: str
    content: str
    status: str # "pending", "completed", "error"
    agent_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    question: Optional[str] = None
    answer: Optional[str] = None
    original_content: Optional[str] = None

class RagDocument(BaseModel):
    org_id: Optional[str] = None
    agent_id: Optional[str] = None
    title: str
    content: str
    embedding: Optional[List[float]] = None
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    question: Optional[str] = None
    answer: Optional[str] = None
    original_content: Optional[str] = None
