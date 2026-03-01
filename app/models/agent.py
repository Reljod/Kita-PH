from pydantic import BaseModel, Field
from typing import Any, List, Optional
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
    personalities: Optional[List[str]] = None

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    llm_id: Optional[str] = None
    personalities: Optional[List[str]] = None

class AgentResponse(BaseModel):
    id: str
    base_id: Optional[str] = None
    version: int
    name: str
    role: str
    goal: str
    backstory: str
    personalities: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    llm_id: str
    last_chat: Optional[Any] = None
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
    personalities: Optional[List[str]] = None
    llm_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

def format_agent_response(doc: dict, system_prompt: Optional[str] = None) -> AgentResponse:
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
        personalities=doc.get("personalities"),
        system_prompt=system_prompt,
        llm_id=doc["llm_id"],
        created_at=doc.get("created_at", datetime.utcnow()),
        updated_at=doc.get("updated_at", datetime.utcnow())
    )
