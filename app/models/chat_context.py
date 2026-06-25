from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime, timezone


class ChatContextEntry(BaseModel):
    chat_id: str
    key: str
    value: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatArchive(BaseModel):
    chat_id: str
    org_id: Optional[str] = None
    turn_range: tuple[int, int]
    messages: list[Any]
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
