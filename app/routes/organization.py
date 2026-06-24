import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List
from app.models.organization import (
    OrganizationResponse, OrgCreate, OrgUpdate, OrgMemberUpdate, OrgIntegrationUpdate
)
from app.models.user import UserResponse
from app.security import get_current_user, require_org_membership
from app.dependencies import get_org_service
from app.services.organization_service import OrganizationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/org", tags=["org"])

async def run_org_scaffolding(org_id: str):
    logger.info(f"Starting organization scaffolding for org_id: {org_id}")
    from app.services.llm_service import LlmService
    from app.services.agent_service import AgentService
    from app.services.tool_service import ToolService
    from app.services.rag_service import MongoVectorDbRagService
    from app.services.organization_creation_service import OrganizationCreationService
    from app.services.organization_service import OrganizationService
    from app.services.web_search_service import SerperSearchService
    from app.db import db, TenantCollection

    llm_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_coll)

    agent_coll = TenantCollection(db.get_agents_collection(), org_id)
    tools_coll = TenantCollection(db.get_tools_collection(), org_id)
    agent_service = AgentService(llm_service=llm_service, collection=agent_coll, tools_collection=tools_coll)

    web_search_service = SerperSearchService()
    tool_service = ToolService(web_search_service=web_search_service, collection=tools_coll)

    rag_coll = TenantCollection(db.get_rag_collection(), org_id)
    rag_service = MongoVectorDbRagService(rag_coll)

    org_service = OrganizationService()

    creation_service = OrganizationCreationService(
        llm_service=llm_service,
        agent_service=agent_service,
        tool_service=tool_service,
        rag_service=rag_service,
        org_service=org_service
    )

    try:
        await creation_service.initialize_org(org_id)
        logger.info(f"Successfully completed organization scaffolding for org_id: {org_id}")
    except Exception as e:
        logger.error(f"Error running organization scaffolding for {org_id}: {e}", exc_info=True)

@router.post("/", response_model=OrganizationResponse)
async def create_organization(
    org_in: OrgCreate,
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    logger.info(f"Attempting to create organization: {org_in.org_name} by user: {current_user.id}")
    org = org_service.create_org(org_in, current_user.id)
    logger.info(f"Created organization {org.id}. Queueing scaffolding task.")
    background_tasks.add_task(run_org_scaffolding, org.id)
    return org

@router.get("/{id}/status")
async def get_org_creation_status(
    id: str,
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
):
    # Try finding by ID first, then code
    org = org_service.get_org(id)
    if not org:
        org = org_service.get_org_by_code(id)
        
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
        
    # Check if user is a member of the organization
    is_member = any(member.user_id == current_user.id for member in org.org_members)
    if not is_member:
        raise HTTPException(status_code=403, detail="Access denied to this organization")
        
    return {
        "org_id": org.id,
        "status": org.status or "completed"
    }


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
             logger.warning(f"Access denied adding member to org {id} by user {current_user.id}")
             raise HTTPException(status_code=403, detail="Access denied to this organization")

    logger.info(f"Adding/updating member {member_in.user_id} with role {member_in.role} in org {authorized_org_id}")
    org = org_service.add_or_update_member(authorized_org_id, member_in)
    if not org:
        logger.warning(f"Org {authorized_org_id} not found for member update")
        raise HTTPException(status_code=404, detail="Organization not found")
    logger.info(f"Successfully added/updated member {member_in.user_id} in org {authorized_org_id}")
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
             logger.warning(f"Access denied revoking member from org {id} by user {current_user.id}")
             raise HTTPException(status_code=403, detail="Access denied to this organization")

    logger.info(f"Revoking member {user_id} from org {authorized_org_id}")
    org = org_service.revoke_member(authorized_org_id, user_id)
    if not org:
        logger.warning(f"Org {authorized_org_id} not found for member revocation")
        raise HTTPException(status_code=404, detail="Organization not found")
    logger.info(f"Successfully revoked member {user_id} from org {authorized_org_id}")
    return org
