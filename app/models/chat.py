from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatCreateRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    id: str
    messages: List[Any]
    agent_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class ChatContinueRequest(BaseModel):
    message: str

# Model to represent DB document
class ChatDocument(BaseModel):
    org_id: Optional[str] = None
    messages: List[Any] = Field(default_factory=list)
    agent_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
