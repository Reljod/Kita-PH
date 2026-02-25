from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from app.models.auth import Token, RegisterRequest, LoginRequest
from app.models.user import UserCreate
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.organization_service import OrganizationService
from app.security import get_auth_service, get_user_service, get_org_service, oauth2_scheme

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=Token)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service)
):
    existing_user = user_service.get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_in = UserCreate(
        email=request.email,
        password=request.password,
        first_name=request.first_name,
        last_name=request.last_name
    )
    user = user_service.create_user(user_in)
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
    if not org_id and not org_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization identification (org_id or org_code) is required"
        )

    user = user_service.get_user_by_email(form_data.username)
    if not user or not user_service.verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    resolved_org_id = org_id
    if org_code:
        org = org_service.get_org_by_code(org_code)
        if not org:
            raise HTTPException(status_code=404, detail=f"Organization code '{org_code}' not found")
        resolved_org_id = org.id
    
    # If org_id/org_code provided, verify user is a member
    if resolved_org_id:
        org = org_service.get_org(resolved_org_id)
        if not org:
            raise HTTPException(status_code=404, detail=f"Organization with ID '{resolved_org_id}' not found")
        is_member = any(m.user_id == str(user["_id"]) for m in org.org_members)
        if not is_member:
            raise HTTPException(status_code=403, detail="User is not a member of this organization")

    return auth_service.generate_tokens(str(user["_id"]), resolved_org_id)

@router.post("/refresh", response_model=Token)
async def refresh(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    token_data = auth_service.verify_token(refresh_token)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Revoke old tokens
    auth_service.revoke_token(refresh_token)
    
    return auth_service.generate_tokens(token_data.user_id)

@router.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service)
):
    auth_service.revoke_token(token)
    return {"message": "Successfully logged out"}
