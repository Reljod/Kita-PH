import os
import json
import logging
from typing import Any, Optional
from datetime import datetime, timezone

import httpx

from app.db import TenantCollection

logger = logging.getLogger(__name__)

ARCHIVE_THRESHOLD = int(os.getenv("CHAT_ARCHIVE_THRESHOLD", "20"))
ARCHIVE_TRIGGER = int(os.getenv("CHAT_ARCHIVE_TRIGGER", "40"))
CONTEXT_LLM_MODEL = os.getenv("CONTEXT_LLM_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class ChatArchiveService:
    def __init__(self, collection: TenantCollection):
        self.collection = collection

    def _count_turns(self, messages: list) -> int:
        turn_count = 0
        for msg in messages:
            role = ""
            if isinstance(msg, dict):
                role = msg.get("role", "")
            else:
                role = getattr(msg, "role", "")
            if role == "user":
                turn_count += 1
        return turn_count

    async def _generate_summary(self, messages: list) -> Optional[str]:
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set, skipping summary generation")
            return None
        try:
            serialized = json.dumps(messages, default=str)
            truncated = serialized[:10000]
            prompt = (
                "Summarize the following conversation in 2-3 sentences. "
                "Capture key decisions, user preferences, and important context.\n\n"
                f"{truncated}"
            )
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": CONTEXT_LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate archive summary: {e}")
            return None

    async def archive_old_messages(
        self, chat_id: str, messages: list, org_id: Optional[str] = None
    ) -> tuple[list, Optional[str]]:
        turn_count = self._count_turns(messages)
        if turn_count < ARCHIVE_TRIGGER:
            return messages, None

        keep_count = ARCHIVE_THRESHOLD
        kept_messages = messages[-keep_count:] if len(messages) > keep_count else messages
        archived_messages = messages[:-keep_count] if len(messages) > keep_count else []

        if not archived_messages:
            return messages, None

        summary = await self._generate_summary(archived_messages)

        turn_start = 0
        turn_end = turn_count - keep_count - 1

        archive_doc = {
            "chat_id": chat_id,
            "org_id": org_id,
            "turn_range": [turn_start, turn_end],
            "messages": archived_messages,
            "summary": summary,
            "created_at": datetime.now(timezone.utc),
        }
        self.collection.insert_one(archive_doc)
        logger.info(
            f"Archived messages for chat {chat_id}: "
            f"turns {turn_start}-{turn_end} archived, "
            f"{len(kept_messages)} messages kept"
        )
        return kept_messages, summary

    def get_archived_messages(
        self, chat_id: str, page: int = 1, page_size: int = 50
    ) -> list[dict]:
        cursor = (
            self.collection.find({"chat_id": chat_id})
            .sort("created_at", -1)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        return list(cursor)

    def get_all_summaries(self, chat_id: str) -> list[str]:
        docs = self.collection.find(
            {"chat_id": chat_id, "summary": {"$ne": None}}
        ).sort("created_at", 1)
        summaries = []
        for doc in docs:
            if doc.get("summary"):
                summaries.append(doc["summary"])
        return summaries
