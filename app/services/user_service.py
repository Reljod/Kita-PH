from typing import Optional
from datetime import datetime, timezone
from bson import ObjectId
import bcrypt
from app.db import Database
from app.models.user import UserDocument, UserCreate, UserUpdate, PasswordUpdate, UserResponse

class UserService:
    def __init__(self):
        self.collection = Database.get_users_collection()

    def hash_password(self, password: str) -> str:
        # Encode password to bytes
        password_bytes = password.encode('utf-8')
        # Generate salt and hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        # Return as string for database storage
        return hashed.decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        # Encode both to bytes
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        # Verify
        return bcrypt.checkpw(password_bytes, hashed_bytes)

    def create_user(self, user_in: UserCreate) -> UserResponse:
        hashed_password = self.hash_password(user_in.password)
        user_dict = user_in.model_dump()
        user_dict["password"] = hashed_password
        user_dict["created_at"] = datetime.now(timezone.utc)
        user_dict["updated_at"] = datetime.now(timezone.utc)
        
        result = self.collection.insert_one(user_dict)
        user_dict["id"] = str(result.inserted_id)
        return UserResponse(**user_dict)

    def get_user_by_email(self, email: str) -> Optional[dict]:
        user = self.collection.find_one({"email": email})
        if user:
            user["id"] = str(user["_id"])
        return user

    def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        user = self.collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user["id"] = str(user["_id"])
            return UserResponse(**user)
        return None

    def update_user(self, user_id: str, user_in: UserUpdate) -> Optional[UserResponse]:
        update_data = user_in.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        result = self.collection.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$set": update_data},
            return_document=True
        )
        if result:
            result["id"] = str(result["_id"])
            return UserResponse(**result)
        return None

    def update_password(self, user_id: str, password_in: PasswordUpdate) -> bool:
        user = self.collection.find_one({"_id": ObjectId(user_id)})
        if not user or not self.verify_password(password_in.old_password, user["password"]):
            return False
        
        hashed_password = self.hash_password(password_in.new_password)
        self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"password": hashed_password, "updated_at": datetime.now(timezone.utc)}}
        )
        return True
