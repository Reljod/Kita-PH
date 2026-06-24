from datetime import datetime, timezone
from typing import Optional
import os

from app.db import TenantCollection
from app.services.redis_service import RedisService


REDIS_TTL_SECONDS = int(os.getenv("CHAT_CONTEXT_REDIS_TTL", "3600"))
REDIS_KEY_PREFIX = "chat_context"


class ChatContextService:
    def __init__(self, collection: TenantCollection):
        self.collection = collection
        self.redis = RedisService.get_client()

    def _redis_key(self, chat_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:{chat_id}"

    async def store_fact(self, chat_id: str, key: str, value: str) -> None:
        now = datetime.now(timezone.utc)
        self.collection.update_one(
            {"chat_id": chat_id, "key": key},
            {"$set": {"value": value, "updated_at": now}},
            upsert=True,
        )
        redis_key = self._redis_key(chat_id)
        await self.redis.hset(redis_key, key, value)
        await self.redis.expire(redis_key, REDIS_TTL_SECONDS)

    async def get_facts(self, chat_id: str) -> dict[str, str]:
        redis_key = self._redis_key(chat_id)
        cached = await self.redis.hgetall(redis_key)
        if cached:
            return cached
        docs = list(self.collection.find({"chat_id": chat_id}))
        facts = {doc["key"]: doc["value"] for doc in docs}
        if facts:
            await self.redis.hset(redis_key, mapping=facts)
            await self.redis.expire(redis_key, REDIS_TTL_SECONDS)
        return facts

    async def get_fact(self, chat_id: str, key: str) -> Optional[str]:
        redis_key = self._redis_key(chat_id)
        cached = await self.redis.hget(redis_key, key)
        if cached is not None:
            return cached
        doc = self.collection.find_one({"chat_id": chat_id, "key": key})
        if doc:
            await self.redis.hset(redis_key, key, doc["value"])
            await self.redis.expire(redis_key, REDIS_TTL_SECONDS)
            return doc["value"]
        return None

    async def delete_fact(self, chat_id: str, key: str) -> None:
        self.collection.delete_one({"chat_id": chat_id, "key": key})
        redis_key = self._redis_key(chat_id)
        await self.redis.hdel(redis_key, key)

    async def format_facts_for_prompt(self, chat_id: str) -> str:
        facts = await self.get_facts(chat_id)
        if not facts:
            return ""
        lines = ["Key facts:"]
        for key, value in facts.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
