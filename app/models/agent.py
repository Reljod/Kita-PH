from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

def parse_agent_id(agent_id_str: str) -> tuple[str, Optional[int]]:
    if "-v" in agent_id_str:
        base, ver = agent_id_str.rsplit("-v", 1)
        if ver.isdigit():
            return base, int(ver)
    return agent_id_str, None

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
    org_id: Optional[str] = None
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

def format_agent_response(doc: dict) -> AgentResponse:
    base_id = doc.get("base_id") or str(doc["_id"])
    version = doc.get("version", 1)
    
    formatted_id = f"{base_id}-v{version}" if version > 1 else base_id

    return AgentResponse(
        id=formatted_id,
        base_id=base_id,
        version=version,
        name=doc["name"],
        role=doc["role"],
        goal=doc["goal"],
        backstory=doc["backstory"],
        system_prompt=doc.get("system_prompt"),
        status=doc.get("status", "completed"),
        llm_id=doc["llm_id"],
        created_at=doc.get("created_at", datetime.utcnow()),
        updated_at=doc.get("updated_at", datetime.utcnow())
    )
