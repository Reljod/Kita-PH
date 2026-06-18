from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone
from enum import Enum

class OrgRole(str, Enum):
    ADMIN = "ADMIN"
    DEV = "DEV"
    MEMBER = "MEMBER"

class OrgMember(BaseModel):
    user_id: str
    role: OrgRole

class Integrations(BaseModel):
    facebook_page_id: Optional[str] = None

class OrganizationBase(BaseModel):
    org_name: str = ""
    org_code: str = ""
    integrations: Optional[Integrations] = Field(default_factory=Integrations)
    status: Optional[str] = "completed"


class OrgCreate(BaseModel):
    org_name: str = Field(..., min_length=1, max_length=100)
    org_code: str = Field(..., min_length=1, max_length=100)

class OrgUpdate(BaseModel):
    org_name: Optional[str] = Field(None, min_length=1, max_length=100)
    org_code: Optional[str] = Field(None, min_length=1, max_length=100)

class OrgMemberUpdate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    role: OrgRole

class OrgIntegrationUpdate(BaseModel):
    facebook_page_id: Optional[str] = Field(None, max_length=100)

class OrganizationResponse(OrganizationBase):
    id: str
    org_members: List[OrgMember] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class OrganizationDocument(OrganizationBase):
    org_members: List[OrgMember] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
