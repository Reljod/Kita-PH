from pydantic import BaseModel, Field
from typing import Any, List, Optional
from datetime import datetime, timezone

def parse_agent_id(agent_id_str: str) -> tuple[str, Optional[int]]:
    if "-v" in agent_id_str:
        base, ver = agent_id_str.rsplit("-v", 1)
        if ver.isdigit():
            return base, int(ver)
    return agent_id_str, None

class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., min_length=1, max_length=200)
    goal: str = Field(..., min_length=1, max_length=1000)
    backstory: str = Field(..., min_length=1, max_length=2000)
    llm_id: str = Field(..., min_length=1, max_length=100)
    personalities: Optional[List[str]] = Field(None, max_length=50)
    tools: Optional[List[str]] = Field(default_factory=list, max_length=50)

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[str] = Field(None, min_length=1, max_length=200)
    goal: Optional[str] = Field(None, min_length=1, max_length=1000)
    backstory: Optional[str] = Field(None, min_length=1, max_length=2000)
    llm_id: Optional[str] = Field(None, min_length=1, max_length=100)
    personalities: Optional[List[str]] = Field(None, max_length=50)
    tools: Optional[List[str]] = Field(None, max_length=50)
    config: Optional[Dict[str, Any]] = None

class AddToolsRequest(BaseModel):
    tool_ids: List[str] = Field(..., min_length=1, max_length=50)

class RemoveToolsRequest(BaseModel):
    tool_ids: List[str] = Field(..., min_length=1, max_length=50)

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
    tools: Optional[List[str]] = Field(default_factory=list)
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
    tools: Optional[List[str]] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

def format_agent_response(doc: dict, system_prompt: Optional[str] = None) -> AgentResponse:
    base_id = doc.get("base_id") or str(doc["_id"])
    version = doc.get("version", 1)
    
    formatted_id = f"{base_id}-v{version}" if version > 1 else base_id

    return AgentResponse(
        id=base_id,
        base_id=base_id,
        version=version,
        name=doc["name"],
        role=doc["role"],
        goal=doc["goal"],
        backstory=doc["backstory"],
        personalities=doc.get("personalities"),
        system_prompt=system_prompt,
        llm_id=doc["llm_id"],
        tools=doc.get("tools", []),
        created_at=doc.get("created_at", datetime.now(timezone.utc)),
        updated_at=doc.get("updated_at", datetime.now(timezone.utc))
    )

