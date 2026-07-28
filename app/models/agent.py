from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from datetime import datetime, timezone


class AgentLanguage(str, Enum):
    """The language an agent speaks in.

    A closed enum rather than a free-text field: the value selects a fixed,
    repo-owned instruction block, so nothing a user types ever reaches the
    system prompt through this route.
    """

    ENGLISH = "english"
    FILIPINO = "filipino"


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
    language: AgentLanguage = AgentLanguage.ENGLISH
    tools: Optional[List[str]] = Field(default_factory=list, max_length=50)


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[str] = Field(None, min_length=1, max_length=200)
    goal: Optional[str] = Field(None, min_length=1, max_length=1000)
    backstory: Optional[str] = Field(None, min_length=1, max_length=2000)
    llm_id: Optional[str] = Field(None, min_length=1, max_length=100)
    personalities: Optional[List[str]] = Field(None, max_length=50)
    language: Optional[AgentLanguage] = None
    tools: Optional[List[str]] = Field(None, max_length=50)


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
    language: AgentLanguage = AgentLanguage.ENGLISH
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
    language: AgentLanguage = AgentLanguage.ENGLISH
    llm_id: str
    tools: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def resolve_agent_language(value: Any) -> AgentLanguage:
    """Coerce a stored language value into an AgentLanguage.

    Documents written before this field existed have no `language` key, and
    the collection is not migrated. An unreadable value must not make an
    otherwise-valid agent unloadable, so anything unrecognised reads as
    English — the behaviour those agents already had.
    """
    try:
        return AgentLanguage(value)
    except ValueError:
        return AgentLanguage.ENGLISH


def format_agent_response(
    doc: dict, system_prompt: Optional[str] = None
) -> AgentResponse:
    base_id = doc.get("base_id") or str(doc["_id"])
    version = doc.get("version", 1)

    # `id` is deliberately the bare base_id even for later versions — pinning
    # the versioned form here would change every client URL. A vestigial
    # `formatted_id` used to be computed for that and never used; CI lints
    # whole changed files, so it had to go rather than sit here unread.
    return AgentResponse(
        id=base_id,
        base_id=base_id,
        version=version,
        name=doc["name"],
        role=doc["role"],
        goal=doc["goal"],
        backstory=doc["backstory"],
        personalities=doc.get("personalities"),
        language=resolve_agent_language(doc.get("language")),
        system_prompt=system_prompt,
        llm_id=doc["llm_id"],
        tools=doc.get("tools", []),
        created_at=doc.get("created_at", datetime.now(timezone.utc)),
        updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
    )
