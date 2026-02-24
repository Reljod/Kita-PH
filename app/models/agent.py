from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class AgentCreateRequest(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str
    llm_id: str

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    llm_id: Optional[str] = None
    system_prompt: Optional[str] = None
    status: Optional[str] = None

class AgentResponse(BaseModel):
    id: str
    base_id: Optional[str] = None
    version: int
    name: str
    role: str
    goal: str
    backstory: str
    system_prompt: Optional[str]
    status: str
    llm_id: str
    created_at: datetime
    updated_at: datetime

class AgentDocument(BaseModel):
    base_id: Optional[str] = None
    version: int = 1
    name: str
    role: str
    goal: str
    backstory: str
    system_prompt: Optional[str] = None
    status: str = "pending"
    llm_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
