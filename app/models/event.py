from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Optional
from enum import Enum

class EventKey(str, Enum):
    FILE_COMPLETED = "file:completed"
    PARSE_COMPLETED = "parse:completed"

class ParseInput(BaseModel):
    file_id: str
    org_id: str

class IngestInput(BaseModel):
    file_id: str
    org_id: str

class EventPushRequest(BaseModel):
    event_key: EventKey
    payload: Dict[str, Any]
