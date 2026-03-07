from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

class RagCreateRequest(BaseModel):
    title: str
    content: str
    agent_id: Optional[str] = None

class RagUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class RagResponse(BaseModel):
    id: str
    title: str
    content: str
    status: str # "pending", "completed", "error"
    agent_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class RagDocument(BaseModel):
    org_id: Optional[str] = None
    agent_id: Optional[str] = None
    title: str
    content: str
    embedding: Optional[List[float]] = None
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
