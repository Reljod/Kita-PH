import os
import time
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.db import db
from app.services.redis_service import RedisService
from app.services.encryption import decrypt_key

class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Bypass CORS OPTIONS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # 2. Bypass webhooks and API docs
        path = request.url.path
        if (
            path.startswith("/webhook") or
            path.startswith("/docs") or
            path.startswith("/openapi.json") or
            path == "/redoc"
        ):
            return await call_next(request)

        # 3. Retrieve client credentials from headers
        api_key = request.headers.get("x-api-key")
        client_id = request.headers.get("x-client-id")

        # Get the client IP address (handling proxy headers)
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        # Initialize Redis client
        try:
            redis_client = RedisService.get_client()
        except Exception as e:
            # Fallback in case Redis is not available
            print(f"Redis initialization failed in middleware: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Server configuration error: Cache unavailable"}
            )

        # Helper to apply rate limiting and return 401/429
        async def handle_unauthorized(detail_msg: str):
            current_minute = int(time.time() // 60)
            rate_limit_key = f"rate_limit:{ip}:{current_minute}"
            
            try:
                # Increment requests count for this IP in the current minute
                count = await redis_client.incr(rate_limit_key)
                if count == 1:
                    # Set TTL to 60 seconds on first creation
                    await redis_client.expire(rate_limit_key, 60)
                
                if count > 10:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too Many Requests. Rate limit exceeded for unauthenticated requests."}
                    )
            except Exception as e:
                print(f"Redis rate limiting failed: {e}")
                # Continue if Redis fails to increment to avoid blocking legitimate errors

            return JSONResponse(
                status_code=401,
                content={"detail": detail_msg}
            )

        # 4. Check for missing headers
        if not api_key or not client_id:
            return await handle_unauthorized("Missing x-api-key or x-client-id headers")

        # 5. Validate client credentials
        encrypted_api_key = None
        redis_key = f"client:{client_id}"

        try:
            # Check Redis cache first
            encrypted_api_key = await redis_client.get(redis_key)
        except Exception as e:
            print(f"Redis get failed in middleware: {e}")

        if not encrypted_api_key:
            # Fall back to MongoDB (kita_admin.clients)
            try:
                # db.client is pymongo MongoClient
                if not db.client:
                    return JSONResponse(
                        status_code=500,
                        content={"detail": "Server error: Database connection not initialized"}
                    )
                clients_collection = db.client["kita_admin"]["clients"]
                client_doc = clients_collection.find_one({"client_id": client_id})
                
                if not client_doc:
                    return await handle_unauthorized("Invalid x-client-id or x-api-key")
                
                encrypted_api_key = client_doc["_id"]
                
                # Cache the key in Redis (1-hour TTL)
                try:
                    await redis_client.set(redis_key, encrypted_api_key, ex=3600)
                except Exception as e:
                    print(f"Redis set failed in middleware: {e}")
                    
            except Exception as e:
                print(f"MongoDB validation failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Database error during authentication"}
                )

        # 6. Decrypt and verify API key
        try:
            decrypted_api_key = decrypt_key(encrypted_api_key)
            if decrypted_api_key != api_key:
                return await handle_unauthorized("Invalid x-client-id or x-api-key")
        except Exception as e:
            print(f"Decryption failed in middleware: {e}")
            return await handle_unauthorized("Invalid x-client-id or x-api-key")

        # 7. Authentication successful, proceed to route
        return await call_next(request)
