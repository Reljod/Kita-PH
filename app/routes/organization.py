from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.models.organization import (
    OrganizationResponse, OrgCreate, OrgUpdate, OrgMemberUpdate, OrgIntegrationUpdate
)
from app.models.user import UserResponse
from app.models.user import UserResponse
from app.security import get_current_user, require_org_membership
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
    org_service: OrganizationService = Depends(get_org_service),
    authorized_org_id: str = Depends(require_org_membership)
):
    # Ensure the user is accessing the organization they are logged into
    if id != authorized_org_id:
        # Fallback check if it's an org_code
        org = org_service.get_org_by_code(id)
        if not org or org.id != authorized_org_id:
             raise HTTPException(status_code=403, detail="Access denied to this organization")
    
    org = org_service.get_org(authorized_org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.patch("/{id}", response_model=OrganizationResponse)
async def update_organization(
    id: str,
    org_in: OrgUpdate,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    authorized_org_id: str = Depends(require_org_membership)
):
    if id != authorized_org_id:
        org = org_service.get_org_by_code(id)
        if not org or org.id != authorized_org_id:
             raise HTTPException(status_code=403, detail="Access denied to this organization")

    org = org_service.update_org(authorized_org_id, org_in)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.patch("/{id}/integrations", response_model=OrganizationResponse)
async def update_organization_integrations(
    id: str,
    integration_in: OrgIntegrationUpdate,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    authorized_org_id: str = Depends(require_org_membership)
):
    if id != authorized_org_id:
        org = org_service.get_org_by_code(id)
        if not org or org.id != authorized_org_id:
             raise HTTPException(status_code=403, detail="Access denied to this organization")

    org = org_service.update_integrations(authorized_org_id, integration_in)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.put("/{id}/member", response_model=OrganizationResponse)
async def add_or_update_member(
    id: str,
    member_in: OrgMemberUpdate,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    authorized_org_id: str = Depends(require_org_membership)
):
    if id != authorized_org_id:
        org = org_service.get_org_by_code(id)
        if not org or org.id != authorized_org_id:
             raise HTTPException(status_code=403, detail="Access denied to this organization")

    org = org_service.add_or_update_member(authorized_org_id, member_in)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.delete("/{id}/member/{user_id}", response_model=OrganizationResponse)
async def revoke_member(
    id: str,
    user_id: str,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service),
    authorized_org_id: str = Depends(require_org_membership)
):
    if id != authorized_org_id:
        org = org_service.get_org_by_code(id)
        if not org or org.id != authorized_org_id:
             raise HTTPException(status_code=403, detail="Access denied to this organization")

    org = org_service.revoke_member(authorized_org_id, user_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org
