from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

class FileStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

class FileUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    size: int = Field(..., gt=0, le=52428800)  # Size in bytes (max 51,200 KB / 50MB)
    content_type: Optional[str] = Field(None, max_length=100)
    agent_id: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None

class FileUpdateRequest(BaseModel):
    filename: Optional[str] = Field(None, min_length=1, max_length=255)
    agent_id: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[FileStatus] = None

class BatchFileCompleteRequest(BaseModel):
    file_ids: List[str] = Field(..., min_length=1, max_length=100)

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
    status: FileStatus
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
    status: FileStatus = FileStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
