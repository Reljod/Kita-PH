import asyncio
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
        # Synchronous bcrypt hash — only use from sync contexts.
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')

    async def hash_password_async(self, password: str) -> str:
        # bcrypt.gensalt + bcrypt.hashpw are CPU-bound. Offload to thread pool.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.hash_password, password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        # Synchronous bcrypt check — only use from sync contexts.
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)

    async def verify_password_async(self, plain_password: str, hashed_password: str) -> bool:
        # bcrypt.checkpw is CPU-bound and blocks the event loop for hundreds of
        # milliseconds. Running it in a thread-pool executor keeps the async
        # event loop free, preventing Cloudflare 502/524 timeout errors.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            self.verify_password,
            plain_password,
            hashed_password
        )

    def create_user(self, user_in: UserCreate, prehashed: bool = False) -> UserResponse:
        # If prehashed=True, the password was already hashed asynchronously upstream.
        # Otherwise hash it synchronously here (only safe in sync context).
        hashed_password = user_in.password if prehashed else self.hash_password(user_in.password)
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
