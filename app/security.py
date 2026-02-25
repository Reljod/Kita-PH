from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.organization_service import OrganizationService
from app.models.user import UserResponse
from app.models.auth import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_auth_service():
    return AuthService()

def get_user_service():
    return UserService()

def get_org_service():
    return OrganizationService()

async def get_token_data(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service)
) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = auth_service.verify_token(token)
    if token_data is None:
        raise credentials_exception
    return token_data

async def get_current_user(
    token_data: TokenData = Depends(get_token_data),
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    user = user_service.get_user_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_org_id(
    token_data: TokenData = Depends(get_token_data)
) -> str:
    if not token_data.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization context required. Please log in with an organization.",
        )
    return token_data.org_id

async def require_org_membership(
    org_id: str = Depends(get_current_org_id),
    current_user: UserResponse = Depends(get_current_user),
    org_service: OrganizationService = Depends(get_org_service)
) -> str:
    org = org_service.get_org(org_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    is_member = any(member.user_id == current_user.id for member in org.org_members)
    if not is_member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a member of this organization"
        )
    return org_id
