from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

class FileUploadRequest(BaseModel):
    filename: str
    size: int
    content_type: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class FileUpdateRequest(BaseModel):
    filename: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class FileUploadResponse(BaseModel):
    file_id: str
    upload_url: str
    method: str  # "POST" or "TUS"
    token: Optional[str] = None # For signed URLs

class FileResponse(BaseModel):
    id: str
    filename: str
    extension: str
    size: int
    content_type: Optional[str] = None
    org_id: str
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

class FileDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    extension: str
    size: int
    content_type: Optional[str] = None
    org_id: str
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
