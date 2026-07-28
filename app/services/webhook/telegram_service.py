import hmac
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic_core import to_jsonable_python
from pydantic_ai import ModelMessagesTypeAdapter

from app.db import Database
from app.exceptions import (
    IntegrationCredentialError,
    IntegrationNotConfiguredError,
    IntegrationProviderError,
    KitaValidationError,
    TelegramThreadNotFoundError,
)
from app.models.telegram import (
    TelegramApiMessage,
    TelegramConnectRequest,
    TelegramDirection,
    TelegramIntegrationDocument,
    TelegramIntegrationResponse,
    TelegramIntegrationUpdate,
    TelegramMessageDocument,
    TelegramMessageResponse,
    TelegramSender,
    TelegramStatus,
    TelegramThreadDocument,
    TelegramThreadResponse,
    TelegramThreadUpdate,
    TelegramUpdate,
)
from app.services.webhook.telegram_client import (
    ITelegramClient,
    TelegramApiError,
    TelegramClient,
)
from app.utils.logger import set_logging_context

logger = logging.getLogger(__name__)

# Keeping every update id a bot has ever seen would grow the integration
# document without bound. Telegram only retries an update for a short window,
# so a bounded recent-window is enough to make delivery idempotent.
PROCESSED_UPDATE_WINDOW = 200


def mask_token(bot_token: str) -> str:
    """Render a bot token safe to return over the API.

    Telegram tokens are `<bot_id>:<secret>`; the bot id is public (it is the
    bot's user id) so it stays readable, and only the tail of the secret is
    kept so a human can confirm which token is installed.
    """
    prefix, _, secret = bot_token.partition(":")
    if not secret:
        return "•" * 8
    return f"{prefix}:{'•' * 8}{secret[-4:]}"


def thread_display_name(doc: Dict[str, Any]) -> str:
    if doc.get("title"):
        return doc["title"]
    names = [doc.get("first_name"), doc.get("last_name")]
    full = " ".join(n for n in names if n).strip()
    if full:
        return full
    if doc.get("username"):
        return f"@{doc['username']}"
    return f"Chat {doc.get('telegram_chat_id')}"


def format_thread_response(doc: Dict[str, Any]) -> TelegramThreadResponse:
    return TelegramThreadResponse(
        id=str(doc["_id"]),
        telegram_chat_id=doc["telegram_chat_id"],
        chat_type=doc.get("chat_type", "private"),
        title=doc.get("title"),
        username=doc.get("username"),
        first_name=doc.get("first_name"),
        last_name=doc.get("last_name"),
        display_name=thread_display_name(doc),
        auto_reply=doc.get("auto_reply"),
        unread_count=doc.get("unread_count", 0),
        last_message_at=doc.get("last_message_at"),
        last_message_preview=doc.get("last_message_preview"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


def format_message_response(doc: Dict[str, Any]) -> TelegramMessageResponse:
    return TelegramMessageResponse(
        id=str(doc["_id"]),
        thread_id=doc["thread_id"],
        direction=doc["direction"],
        sender=doc["sender"],
        text=doc.get("text", ""),
        telegram_message_id=doc.get("telegram_message_id"),
        created_at=doc["created_at"],
    )


class TelegramService:
    """Owns the Telegram channel: connecting a bot, receiving updates, and
    replying — either from the org's agent or from a human in the inbox.

    Collections are held raw rather than through `TenantCollection` because
    the webhook path is by definition pre-tenant: an inbound POST identifies
    itself with a `webhook_id`, and resolving that to an org is the first
    thing this service does. Every method that runs *after* that point takes
    `org_id` explicitly and filters on it.
    """

    def __init__(self, client: Optional[ITelegramClient] = None):
        self.client = client or TelegramClient()

    # --- collections (resolved lazily so tests can patch Database) ---

    @property
    def integrations(self):
        return Database.get_telegram_integrations_collection()

    @property
    def threads(self):
        return Database.get_telegram_threads_collection()

    @property
    def messages(self):
        return Database.get_telegram_messages_collection()

    # --- configuration ---

    def _webhook_base_url(self) -> str:
        base = os.getenv("TELEGRAM_WEBHOOK_BASE_URL", "").strip().rstrip("/")
        if not base:
            raise KitaValidationError(
                "TELEGRAM_WEBHOOK_BASE_URL is not set. Telegram only delivers to a "
                "public HTTPS URL, so the API needs to know its own public address "
                "before a bot can be connected."
            )
        return base

    def _decrypt_token(self, doc: Dict[str, Any]) -> str:
        from app.services.encryption import decrypt_key

        return decrypt_key(doc["bot_token_encrypted"])

    def ensure_indexes(self) -> None:
        """Create the indexes this integration depends on.

        Called from `connect`, which is the only moment a Telegram document
        first appears, rather than from application startup: creating indexes
        for every optional integration on every boot would slow startup for
        the orgs that use none of them. `create_index` is idempotent, so
        repeating it on reconnect costs nothing.
        """
        try:
            self.integrations.create_index("webhook_id", unique=True)
            self.integrations.create_index("org_id", unique=True)
            self.threads.create_index(
                [("org_id", 1), ("telegram_chat_id", 1)], unique=True
            )
            self.threads.create_index([("org_id", 1), ("last_message_at", -1)])
            self.messages.create_index(
                [("org_id", 1), ("thread_id", 1), ("created_at", 1)]
            )
        except Exception as e:
            # An index that cannot be built should not block an org from
            # connecting; it degrades performance, not correctness — except
            # for the unique thread index, which the upsert filter also
            # enforces logically.
            logger.warning(f"Could not create the Telegram indexes: {e}")

    # --- connect / configure / disconnect ---

    async def connect(
        self, org_id: str, req: TelegramConnectRequest
    ) -> TelegramIntegrationResponse:
        from app.services.encryption import encrypt_key

        bot_token = req.bot_token.strip()
        self.ensure_indexes()

        # Resolve the base URL before touching Telegram: a misconfigured
        # deployment should fail without having registered a webhook it then
        # has to roll back.
        base_url = self._webhook_base_url()

        try:
            me = await self.client.get_me(bot_token)
        except TelegramApiError as e:
            raise IntegrationCredentialError(
                f"Telegram rejected this bot token: {e.description}",
                {"provider": "telegram"},
            )

        existing = self.integrations.find_one({"org_id": org_id})
        # Reuse the ids on reconnect so an already-published webhook URL keeps
        # working when an org rotates its token.
        webhook_id = (existing or {}).get("webhook_id") or secrets.token_urlsafe(24)
        webhook_secret = (existing or {}).get(
            "webhook_secret"
        ) or secrets.token_urlsafe(32)
        webhook_url = f"{base_url}/webhook/telegram/{webhook_id}"

        try:
            await self.client.set_webhook(bot_token, webhook_url, webhook_secret)
        except TelegramApiError as e:
            raise IntegrationProviderError(
                f"Could not register the Telegram webhook: {e.description}",
                {"provider": "telegram", "webhook_url": webhook_url},
            )

        doc = TelegramIntegrationDocument(
            org_id=org_id,
            bot_token_encrypted=encrypt_key(bot_token),
            bot_id=me.get("id"),
            bot_username=me.get("username"),
            bot_name=me.get("first_name"),
            webhook_id=webhook_id,
            webhook_secret=webhook_secret,
            webhook_url=webhook_url,
            token_hint=mask_token(bot_token),
            agent_id=req.agent_id or (existing or {}).get("agent_id"),
            auto_reply=req.auto_reply,
            status=TelegramStatus.CONNECTED,
        ).model_dump()

        if existing:
            doc.pop("created_at", None)
            # Preserve the dedupe window across a reconnect.
            doc["processed_update_ids"] = existing.get("processed_update_ids", [])
            doc["updated_at"] = datetime.now(timezone.utc)
            self.integrations.update_one({"_id": existing["_id"]}, {"$set": doc})
        else:
            self.integrations.insert_one(doc)

        logger.info(
            f"Connected Telegram bot @{doc.get('bot_username')} to org {org_id}"
        )
        return self._to_response(self.integrations.find_one({"org_id": org_id}))

    def get_integration(self, org_id: str) -> TelegramIntegrationResponse:
        doc = self.integrations.find_one({"org_id": org_id})
        if not doc:
            return TelegramIntegrationResponse(connected=False)
        return self._to_response(doc)

    def update_integration(
        self, org_id: str, req: TelegramIntegrationUpdate
    ) -> TelegramIntegrationResponse:
        doc = self.integrations.find_one({"org_id": org_id})
        if not doc:
            raise IntegrationNotConfiguredError("telegram")

        update_data = req.model_dump(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.now(timezone.utc)
            self.integrations.update_one({"_id": doc["_id"]}, {"$set": update_data})

        return self._to_response(self.integrations.find_one({"_id": doc["_id"]}))

    async def disconnect(self, org_id: str) -> None:
        doc = self.integrations.find_one({"org_id": org_id})
        if not doc:
            raise IntegrationNotConfiguredError("telegram")

        try:
            await self.client.delete_webhook(self._decrypt_token(doc))
        except (TelegramApiError, Exception) as e:
            # A bot whose token was already revoked upstream must still be
            # removable here, or the org is stuck with a dead integration it
            # cannot replace.
            logger.warning(
                f"Could not delete the Telegram webhook for org {org_id}, "
                f"removing the local record anyway: {e}"
            )

        self.integrations.delete_one({"_id": doc["_id"]})
        logger.info(f"Disconnected the Telegram integration for org {org_id}")

    def _to_response(self, doc: Dict[str, Any]) -> TelegramIntegrationResponse:
        return TelegramIntegrationResponse(
            connected=True,
            bot_id=doc.get("bot_id"),
            bot_username=doc.get("bot_username"),
            bot_name=doc.get("bot_name"),
            masked_token=doc.get("token_hint"),
            webhook_url=doc.get("webhook_url"),
            agent_id=doc.get("agent_id"),
            auto_reply=doc.get("auto_reply", True),
            status=doc.get("status"),
            error_message=doc.get("error_message"),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
        )

    # --- webhook ingress ---

    def resolve_webhook(
        self, webhook_id: str, secret_header: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Return the integration this webhook belongs to, or None if the
        request cannot be authenticated.

        Both failure modes return None rather than distinguishing "unknown
        webhook" from "wrong secret", so a caller probing the endpoint learns
        nothing about which webhook ids exist.
        """
        doc = self.integrations.find_one({"webhook_id": webhook_id})
        if not doc:
            return None

        expected = doc.get("webhook_secret") or ""
        if not secret_header or not hmac.compare_digest(secret_header, expected):
            logger.warning(
                f"Rejected a Telegram webhook delivery for org {doc.get('org_id')}: "
                "the secret token did not match"
            )
            return None

        return doc

    def _claim_update(self, integration_id: Any, update_id: int) -> bool:
        """Record an update id, returning False if it was already seen.

        The `$ne` guard makes the check-and-set a single atomic operation, so
        two concurrent redeliveries of the same update cannot both win. Without
        it, acknowledging Telegram before the agent has answered would turn one
        slow reply into several identical ones.
        """
        result = self.integrations.update_one(
            {"_id": integration_id, "processed_update_ids": {"$ne": update_id}},
            {
                "$push": {
                    "processed_update_ids": {
                        "$each": [update_id],
                        "$slice": -PROCESSED_UPDATE_WINDOW,
                    }
                }
            },
        )
        return result.modified_count > 0

    async def process_update(
        self, integration: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        """Handle one Telegram update end to end. Runs in the background, so
        it swallows its own errors rather than surfacing them to Telegram —
        a raised exception here would only be logged by the task runner."""
        org_id = integration.get("org_id")
        set_logging_context(org_id=org_id)

        try:
            update = TelegramUpdate.model_validate(payload)
        except Exception as e:
            logger.warning(f"Ignoring an unparseable Telegram update: {e}")
            return

        message = update.message or update.edited_message
        if not message or not message.text:
            # Stickers, joins, photos and the like: nothing to answer yet.
            logger.info(
                f"Ignoring Telegram update {update.update_id} with no text message"
            )
            return

        if not self._claim_update(integration["_id"], update.update_id):
            logger.info(
                f"Skipping Telegram update {update.update_id}: already processed"
            )
            return

        try:
            thread = self._upsert_thread(org_id, message)
            self._record_message(
                org_id=org_id,
                thread_id=str(thread["_id"]),
                direction=TelegramDirection.INBOUND,
                sender=TelegramSender.USER,
                text=message.text,
                telegram_message_id=message.message_id,
                bump_unread=True,
            )

            auto_reply = thread.get("auto_reply")
            if auto_reply is None:
                auto_reply = integration.get("auto_reply", True)

            if not auto_reply:
                logger.info(
                    f"Auto-reply is off for Telegram thread {thread['_id']}; "
                    "leaving the message for a human"
                )
                return

            await self._reply_with_agent(integration, thread, message.text)
        except Exception as e:
            logger.error(
                f"Failed to process Telegram update {update.update_id} for org "
                f"{org_id}: {e}",
                exc_info=True,
            )

    def _upsert_thread(
        self, org_id: str, message: TelegramApiMessage
    ) -> Dict[str, Any]:
        chat = message.chat
        now = datetime.now(timezone.utc)
        sender = message.from_user

        profile = {
            "chat_type": chat.type,
            "title": chat.title,
            "username": chat.username or (sender.username if sender else None),
            "first_name": chat.first_name or (sender.first_name if sender else None),
            "last_name": chat.last_name or (sender.last_name if sender else None),
            "updated_at": now,
        }

        defaults = TelegramThreadDocument(
            org_id=org_id, telegram_chat_id=chat.id
        ).model_dump()
        # `$setOnInsert` and `$set` may not touch the same field, so anything
        # $set writes has to come out of the insert defaults.
        for key in list(profile.keys()):
            defaults.pop(key, None)

        self.threads.update_one(
            {"org_id": org_id, "telegram_chat_id": chat.id},
            {"$set": profile, "$setOnInsert": defaults},
            upsert=True,
        )
        return self.threads.find_one({"org_id": org_id, "telegram_chat_id": chat.id})

    def _record_message(
        self,
        org_id: str,
        thread_id: str,
        direction: TelegramDirection,
        sender: TelegramSender,
        text: str,
        telegram_message_id: Optional[int] = None,
        bump_unread: bool = False,
    ) -> Dict[str, Any]:
        doc = TelegramMessageDocument(
            org_id=org_id,
            thread_id=thread_id,
            direction=direction,
            sender=sender,
            text=text,
            telegram_message_id=telegram_message_id,
        ).model_dump()
        result = self.messages.insert_one(doc)
        doc["_id"] = result.inserted_id

        preview = text[:120] + "..." if len(text) > 120 else text
        update: Dict[str, Any] = {
            "$set": {
                "last_message_at": doc["created_at"],
                "last_message_preview": preview,
                "updated_at": doc["created_at"],
            }
        }
        if bump_unread:
            update["$inc"] = {"unread_count": 1}

        self.threads.update_one({"org_id": org_id, "_id": ObjectId(thread_id)}, update)
        return doc

    # --- replying ---

    def _resolve_agent_id(self, integration: Dict[str, Any], org_id: str):
        """Return the agent that should answer, binding the org's first agent
        if none was chosen explicitly.

        Resolution is lazy because an org connects Telegram while scaffolding
        may still be creating its agents; binding at connect time would leave
        the integration pointing at nothing.
        """
        from app.dependencies.services import get_services

        agent_service = get_services(org_id).agent_service

        agent_id = integration.get("agent_id")
        if agent_id:
            return agent_id, agent_service

        agents = agent_service.get_all_agents()
        if not agents:
            return None, agent_service

        agent_id = agents[0].id
        self.integrations.update_one(
            {"_id": integration["_id"]},
            {"$set": {"agent_id": agent_id, "updated_at": datetime.now(timezone.utc)}},
        )
        integration["agent_id"] = agent_id
        logger.info(f"Bound Telegram for org {org_id} to agent {agent_id}")
        return agent_id, agent_service

    async def _reply_with_agent(
        self, integration: Dict[str, Any], thread: Dict[str, Any], query: str
    ) -> None:
        org_id = integration["org_id"]
        agent_id, agent_service = self._resolve_agent_id(integration, org_id)
        if not agent_id:
            logger.warning(
                f"Org {org_id} has a Telegram bot connected but no agent to answer with"
            )
            return

        history = None
        stored_history = thread.get("agent_history") or []
        if stored_history:
            try:
                history = ModelMessagesTypeAdapter.validate_python(stored_history)
            except Exception as e:
                # A history the current pydantic-ai version cannot read should
                # cost this conversation its memory, not its reply.
                logger.warning(
                    f"Discarding unreadable agent history for Telegram thread "
                    f"{thread['_id']}: {e}"
                )

        result = await agent_service.run(
            agent_id=agent_id,
            query=query,
            message_history=history,
            chat_id=str(thread["_id"]),
        )

        answer = self._extract_text(result)
        if not answer:
            logger.warning(
                f"The agent produced no text for Telegram thread {thread['_id']}"
            )
            return

        self.threads.update_one(
            {"_id": thread["_id"]},
            {
                "$set": {
                    "agent_history": to_jsonable_python(result.all_messages()),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

        await self._send(
            integration=integration,
            thread=thread,
            text=answer,
            sender=TelegramSender.AGENT,
        )

    @staticmethod
    def _extract_text(result: Any) -> str:
        output = getattr(result, "output", None)
        if output is None:
            output = getattr(result, "data", None)
        if output is None:
            return ""
        return output if isinstance(output, str) else str(output)

    async def _send(
        self,
        integration: Dict[str, Any],
        thread: Dict[str, Any],
        text: str,
        sender: TelegramSender,
    ) -> Dict[str, Any]:
        bot_token = self._decrypt_token(integration)
        try:
            sent = await self.client.send_message(
                bot_token, thread["telegram_chat_id"], text
            )
        except TelegramApiError as e:
            raise IntegrationProviderError(
                f"Telegram refused the message: {e.description}",
                {"provider": "telegram"},
            )

        return self._record_message(
            org_id=integration["org_id"],
            thread_id=str(thread["_id"]),
            direction=TelegramDirection.OUTBOUND,
            sender=sender,
            text=text,
            telegram_message_id=sent.get("message_id"),
        )

    # --- inbox (org-scoped reads and manual replies) ---

    def list_threads(
        self, org_id: str, limit: int = 50
    ) -> List[TelegramThreadResponse]:
        docs = (
            self.threads.find({"org_id": org_id})
            .sort("last_message_at", -1)
            .limit(limit)
        )
        return [format_thread_response(d) for d in docs]

    def _get_thread(self, org_id: str, thread_id: str) -> Dict[str, Any]:
        try:
            oid = ObjectId(thread_id)
        except Exception:
            raise TelegramThreadNotFoundError(thread_id)

        doc = self.threads.find_one({"org_id": org_id, "_id": oid})
        if not doc:
            raise TelegramThreadNotFoundError(thread_id)
        return doc

    def get_thread(self, org_id: str, thread_id: str) -> TelegramThreadResponse:
        return format_thread_response(self._get_thread(org_id, thread_id))

    def list_messages(
        self, org_id: str, thread_id: str, limit: int = 200
    ) -> List[TelegramMessageResponse]:
        self._get_thread(org_id, thread_id)
        docs = (
            self.messages.find({"org_id": org_id, "thread_id": thread_id})
            .sort("created_at", 1)
            .limit(limit)
        )
        return [format_message_response(d) for d in docs]

    def update_thread(
        self, org_id: str, thread_id: str, req: TelegramThreadUpdate
    ) -> TelegramThreadResponse:
        doc = self._get_thread(org_id, thread_id)
        update_data = req.model_dump(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.now(timezone.utc)
            self.threads.update_one({"_id": doc["_id"]}, {"$set": update_data})
        return format_thread_response(self.threads.find_one({"_id": doc["_id"]}))

    async def send_manual_reply(
        self, org_id: str, thread_id: str, text: str
    ) -> TelegramMessageResponse:
        """Send a message as the bot, written by an org member rather than the
        agent. The agent's history is deliberately left untouched: a human
        reply is not something the agent said, and injecting it would let the
        agent claim authorship of it on the next turn."""
        integration = self.integrations.find_one({"org_id": org_id})
        if not integration:
            raise IntegrationNotConfiguredError("telegram")

        thread = self._get_thread(org_id, thread_id)
        doc = await self._send(
            integration=integration,
            thread=thread,
            text=text,
            sender=TelegramSender.MEMBER,
        )
        return format_message_response(doc)

    def mark_read(self, org_id: str, thread_id: str) -> TelegramThreadResponse:
        doc = self._get_thread(org_id, thread_id)
        self.threads.update_one(
            {"_id": doc["_id"]},
            {"$set": {"unread_count": 0, "updated_at": datetime.now(timezone.utc)}},
        )
        return format_thread_response(self.threads.find_one({"_id": doc["_id"]}))
