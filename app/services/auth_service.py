import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from app.models.auth import Token, TokenData, TokenDocument
from app.db import db

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

class AuthService:
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt, expire

    def create_refresh_token(self, data: dict):
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode = data.copy()
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            org_id: str = payload.get("org_id")
            if user_id is None:
                return None
            
            # Check if token is revoked in database
            token_doc = db.get_tokens_collection().find_one({
                "$or": [
                    {"access_token": token},
                    {"refresh_token": token}
                ],
                "is_revoked": False
            })
            if not token_doc:
                return None

            return TokenData(user_id=user_id, org_id=org_id)
        except JWTError:
            return None

    def generate_tokens(self, user_id: str, org_id: Optional[str] = None) -> Token:
        access_token_data = {"sub": user_id}
        if org_id:
            access_token_data["org_id"] = org_id
            
        access_token, expires_at = self.create_access_token(data=access_token_data)
        refresh_token = self.create_refresh_token(data={"sub": user_id})
        
        # Save tokens to database
        token_doc = TokenDocument(
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at
        )
        db.get_tokens_collection().insert_one(token_doc.dict())

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )

    def revoke_token(self, token: str):
        db.get_tokens_collection().update_many(
            {"$or": [{"access_token": token}, {"refresh_token": token}]},
            {"$set": {"is_revoked": True}}
        )

    def revoke_user_tokens(self, user_id: str):
        db.get_tokens_collection().update_many(
            {"user_id": user_id},
            {"$set": {"is_revoked": True}}
        )

    def revoke_token_pair(self, access_token: str, refresh_token: str):
        db.get_tokens_collection().update_one(
            {"access_token": access_token, "refresh_token": refresh_token},
            {"$set": {"is_revoked": True}}
        )
