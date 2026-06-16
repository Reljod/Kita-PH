import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import UserResponse, UserUpdate, PasswordUpdate
from app.security import get_current_user, get_user_service
from app.services.user_service import UserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_user)):
    logger.info("Retrieved current user profile")
    return current_user

@router.patch("/me", response_model=UserResponse)
async def update_me(
    user_in: UserUpdate,
    current_user: UserResponse = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    logger.info("Attempting user profile update")
    updated_user = user_service.update_user(current_user.id, user_in)
    logger.info("Successfully updated user profile")
    return updated_user

@router.patch("/me/password")
async def update_password(
    password_in: PasswordUpdate,
    current_user: UserResponse = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    logger.info("Attempting user password update")
    success = user_service.update_password(current_user.id, password_in)
    if not success:
        logger.warning("User password update failed: incorrect current password")
        raise HTTPException(status_code=400, detail="Invalid current password")
    logger.info("Successfully updated user password")
    return {"message": "Password updated successfully"}
