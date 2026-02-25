from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.models.organization import (
    OrganizationResponse, OrgCreate, OrgUpdate, OrgMemberUpdate
)
from app.models.user import UserResponse
from app.security import get_current_user
from app.services.organization_service import OrganizationService

router = APIRouter(prefix="/org", tags=["org"])

def get_org_service():
    return OrganizationService()

@router.post("/", response_model=OrganizationResponse)
async def create_organization(
    org_in: OrgCreate,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    return org_service.create_org(org_in, current_user.id)

@router.get("/me", response_model=List[OrganizationResponse])
async def get_my_organizations(
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    return org_service.get_user_orgs(current_user.id)

@router.get("/{id}", response_model=OrganizationResponse)
async def get_organization(
    id: str,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    org = org_service.get_org(id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.patch("/{id}", response_model=OrganizationResponse)
async def update_organization_name(
    id: str,
    org_in: OrgUpdate,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    # Note: In a real app, check if user is admin of this org
    org = org_service.update_org_name(id, org_in)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.put("/{id}/member", response_model=OrganizationResponse)
async def add_or_update_member(
    id: str,
    member_in: OrgMemberUpdate,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    org = org_service.add_or_update_member(id, member_in)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.delete("/{id}/member/{user_id}", response_model=OrganizationResponse)
async def revoke_member(
    id: str,
    user_id: str,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    org = org_service.revoke_member(id, user_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org
