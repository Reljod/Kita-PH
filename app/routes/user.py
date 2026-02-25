from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import UserResponse, UserUpdate, PasswordUpdate
from app.security import get_current_user, get_user_service
from app.services.user_service import UserService

router = APIRouter(prefix="/user", tags=["user"])

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_user)):
    return current_user

@router.patch("/me", response_model=UserResponse)
async def update_me(
    user_in: UserUpdate,
    current_user: UserResponse = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    return user_service.update_user(current_user.id, user_in)

@router.patch("/me/password")
async def update_password(
    password_in: PasswordUpdate,
    current_user: UserResponse = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    success = user_service.update_password(current_user.id, password_in)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid current password")
    return {"message": "Password updated successfully"}
