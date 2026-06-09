import os
import redis.asyncio as aioredis
from typing import Optional

class RedisService:
    _client: Optional[aioredis.Redis] = None

    @classmethod
    def get_client(cls) -> aioredis.Redis:
        if cls._client is None:
            connection_string = os.getenv("REDIS_CONNECTION_STRING")
            if not connection_string:
                raise ValueError("REDIS_CONNECTION_STRING is not set in environment variables")
            cls._client = aioredis.from_url(
                connection_string,
                decode_responses=True
            )
        return cls._client

    @classmethod
    async def close(cls):
        if cls._client:
            await cls._client.aclose()
            cls._client = None
