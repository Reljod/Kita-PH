import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from app.models.auth import Token, RegisterRequest, LoginRequest
from app.models.user import UserCreate
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.organization_service import OrganizationService
from app.security import get_auth_service, get_user_service, get_org_service, oauth2_scheme

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=Token)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service)
):
    logger.info(f"Attempting registration for email: {request.email}")
    existing_user = user_service.get_user_by_email(request.email)
    if existing_user:
        logger.warning(f"Registration failed: email {request.email} already exists")
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_in = UserCreate(
        email=request.email,
        password=request.password,
        first_name=request.first_name,
        last_name=request.last_name
    )
    user = user_service.create_user(user_in)
    logger.info(f"Successfully registered user with email: {request.email}, user_id: {user.id}")
    return auth_service.generate_tokens(user.id)

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    org_id: Optional[str] = Form(None),
    org_code: Optional[str] = Form(None),
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service),
    org_service: OrganizationService = Depends(get_org_service)
):
    logger.info(f"Attempting login for email: {form_data.username}")
    # 1. Authenticate user
    user = user_service.get_user_by_email(form_data.username)
    if not user or not user_service.verify_password(form_data.password, user["password"]):
        logger.warning(f"Login failed: invalid credentials for email: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = str(user["_id"])
    user_orgs = org_service.get_user_orgs(user_id)
    
    resolved_org_id = None
    
    # 2. Handle Organization Identification
    if org_id or org_code:
        # Resolve target org_id
        if org_code:
            org = org_service.get_org_by_code(org_code)
            if not org:
                logger.warning(f"Login failed: org code '{org_code}' not found")
                raise HTTPException(status_code=404, detail=f"Organization code '{org_code}' not found")
            resolved_org_id = org.id
        else:
            resolved_org_id = org_id
            
        # Verify organization exists and user is a member
        target_org = org_service.get_org(resolved_org_id)
        if not target_org:
            logger.warning(f"Login failed: org {resolved_org_id} not found")
            raise HTTPException(status_code=404, detail="Organization not found")
            
        is_member = any(m.user_id == user_id for m in target_org.org_members)
        if not is_member:
            logger.warning(f"Login failed: user {user_id} is not a member of org {resolved_org_id}")
            raise HTTPException(status_code=403, detail="User is not a member of this organization")
    else:
        # No org identification provided
        if user_orgs:
            # User is in at least one org, so they MUST specify which one to log into
            logger.warning(f"Login failed: missing organization identification for email {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization identification (org_id or org_code) is required"
            )
        # User is in NO orgs, allow login without resolved_org_id
        resolved_org_id = None

    logger.info(f"Successfully logged in user: {user_id} (email: {form_data.username}) for org_id: {resolved_org_id}")
    return auth_service.generate_tokens(user_id, resolved_org_id)

@router.post("/refresh", response_model=Token)
async def refresh(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    logger.info("Attempting token refresh")
    token_data = auth_service.verify_token(refresh_token)
    if not token_data:
        logger.warning("Token refresh failed: invalid or expired refresh token")
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Revoke old tokens
    auth_service.revoke_token(refresh_token)
    logger.info(f"Successfully refreshed token for user_id: {token_data.user_id}, org_id: {token_data.org_id}")
    return auth_service.generate_tokens(token_data.user_id, token_data.org_id)

@router.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service)
):
    logger.info("User logging out")
    auth_service.revoke_token(token)
    return {"message": "Successfully logged out"}
