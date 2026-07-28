import logging
from typing import Any, Dict, Optional, Protocol

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

# Telegram hard-caps a single message at 4096 characters and rejects anything
# longer outright, so an over-long agent answer would be dropped entirely
# rather than truncated. Split instead.
MAX_MESSAGE_LENGTH = 4096


class TelegramApiError(Exception):
    """A non-ok response from the Telegram Bot API."""

    def __init__(self, description: str, error_code: Optional[int] = None):
        self.description = description
        self.error_code = error_code
        super().__init__(description)


class ITelegramClient(Protocol):
    async def get_me(self, bot_token: str) -> Dict[str, Any]: ...
    async def set_webhook(
        self, bot_token: str, url: str, secret_token: str
    ) -> Dict[str, Any]: ...
    async def delete_webhook(self, bot_token: str) -> Dict[str, Any]: ...
    async def send_message(
        self, bot_token: str, chat_id: int, text: str
    ) -> Dict[str, Any]: ...


def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into Telegram-sized chunks, preferring paragraph then line
    boundaries so a split does not land mid-sentence."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        # Prefer the last paragraph break, then the last newline, then a hard cut.
        split_at = window.rfind("\n\n")
        if split_at <= 0:
            split_at = window.rfind("\n")
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class TelegramClient(ITelegramClient):
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    async def _call(
        self, bot_token: str, method: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{TELEGRAM_API_BASE}/bot{bot_token}/{method}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload or {})

        try:
            data = response.json()
        except ValueError:
            raise TelegramApiError(
                f"Telegram returned a non-JSON response for {method} "
                f"(status {response.status_code})",
                response.status_code,
            )

        if not data.get("ok"):
            # Never log the URL: it carries the bot token.
            description = data.get("description", "Unknown Telegram API error")
            logger.warning(f"Telegram API {method} failed: {description}")
            raise TelegramApiError(description, data.get("error_code"))

        return data.get("result", {})

    async def get_me(self, bot_token: str) -> Dict[str, Any]:
        return await self._call(bot_token, "getMe")

    async def set_webhook(
        self, bot_token: str, url: str, secret_token: str
    ) -> Dict[str, Any]:
        return await self._call(
            bot_token,
            "setWebhook",
            {
                "url": url,
                "secret_token": secret_token,
                "allowed_updates": ["message", "edited_message"],
                # A bot moving between environments (local tunnel to staging)
                # would otherwise be handed a backlog of updates addressed to
                # the previous webhook.
                "drop_pending_updates": True,
            },
        )

    async def delete_webhook(self, bot_token: str) -> Dict[str, Any]:
        return await self._call(
            bot_token, "deleteWebhook", {"drop_pending_updates": True}
        )

    async def send_message(
        self, bot_token: str, chat_id: int, text: str
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for chunk in split_message(text):
            result = await self._call(
                bot_token,
                "sendMessage",
                {"chat_id": chat_id, "text": chunk},
            )
        return result
