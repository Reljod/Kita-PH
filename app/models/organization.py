from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class OrgRole(str, Enum):
    ADMIN = "ADMIN"
    DEV = "DEV"
    MEMBER = "MEMBER"

class OrgMember(BaseModel):
    user_id: str
    role: OrgRole

class OrganizationBase(BaseModel):
    org_name: str = ""
    org_code: str = ""

class OrgCreate(BaseModel):
    org_name: str
    org_code: str

class OrgUpdate(BaseModel):
    org_name: Optional[str] = None
    org_code: Optional[str] = None

class OrgMemberUpdate(BaseModel):
    user_id: str
    role: OrgRole

class OrganizationResponse(OrganizationBase):
    id: str
    org_members: List[OrgMember] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class OrganizationDocument(OrganizationBase):
    org_members: List[OrgMember] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
