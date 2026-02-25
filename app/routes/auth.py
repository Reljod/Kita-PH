from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models.auth import Token, RegisterRequest, LoginRequest
from app.models.user import UserCreate
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.security import get_auth_service, get_user_service, oauth2_scheme

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
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service)
):
    user = user_service.get_user_by_email(form_data.username)
    if not user or not user_service.verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return auth_service.generate_tokens(user["id"])

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
