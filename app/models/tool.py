from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ToolRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class ToolResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

def format_tool_response(doc: dict) -> ToolResponse:
    return ToolResponse(
        id=str(doc["_id"]),
        name=doc["name"],
        description=doc["description"],
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at")
    )
