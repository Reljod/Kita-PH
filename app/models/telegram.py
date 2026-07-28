from pydantic import BaseModel, Field
from typing import Any, List, Optional
from datetime import datetime, timezone
from enum import Enum


class TelegramDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class TelegramSender(str, Enum):
    USER = "user"
    AGENT = "agent"
    MEMBER = "member"


class TelegramStatus(str, Enum):
    CONNECTED = "connected"
    ERROR = "error"


# --- Incoming Telegram update payloads ---
# Only the fields we actually consume are modelled. Telegram adds fields
# freely, so these stay permissive rather than rejecting unknown keys.


class TelegramApiUser(BaseModel):
    id: int
    is_bot: bool = False
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None


class TelegramApiChat(BaseModel):
    id: int
    type: str = "private"
    title: Optional[str] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class TelegramApiMessage(BaseModel):
    message_id: int
    date: Optional[int] = None
    text: Optional[str] = None
    chat: TelegramApiChat
    from_user: Optional[TelegramApiUser] = Field(None, alias="from")

    model_config = {"populate_by_name": True}


class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[TelegramApiMessage] = None
    edited_message: Optional[TelegramApiMessage] = None


# --- Requests ---


class TelegramConnectRequest(BaseModel):
    bot_token: str = Field(..., min_length=20, max_length=200)
    agent_id: Optional[str] = Field(None, max_length=100)
    auto_reply: bool = True


class TelegramIntegrationUpdate(BaseModel):
    agent_id: Optional[str] = Field(None, max_length=100)
    auto_reply: Optional[bool] = None


class TelegramThreadUpdate(BaseModel):
    # None means "inherit the integration default" — distinct from an explicit
    # False, so `exclude_unset` is what callers must rely on to clear it.
    auto_reply: Optional[bool] = None
    unread_count: Optional[int] = Field(None, ge=0)


class TelegramSendRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)


# --- Responses ---


class TelegramIntegrationResponse(BaseModel):
    connected: bool
    bot_id: Optional[int] = None
    bot_username: Optional[str] = None
    bot_name: Optional[str] = None
    masked_token: Optional[str] = None
    webhook_url: Optional[str] = None
    agent_id: Optional[str] = None
    auto_reply: bool = True
    status: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TelegramThreadResponse(BaseModel):
    id: str
    telegram_chat_id: int
    chat_type: str = "private"
    title: Optional[str] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: str = ""
    auto_reply: Optional[bool] = None
    unread_count: int = 0
    last_message_at: Optional[datetime] = None
    last_message_preview: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TelegramMessageResponse(BaseModel):
    id: str
    thread_id: str
    direction: TelegramDirection
    sender: TelegramSender
    text: str
    telegram_message_id: Optional[int] = None
    created_at: datetime


# --- Documents ---


class TelegramIntegrationDocument(BaseModel):
    org_id: Optional[str] = None
    bot_token_encrypted: str
    bot_id: int
    bot_username: Optional[str] = None
    bot_name: Optional[str] = None
    webhook_id: str
    webhook_secret: str
    webhook_url: Optional[str] = None
    token_hint: str = ""
    agent_id: Optional[str] = None
    auto_reply: bool = True
    status: TelegramStatus = TelegramStatus.CONNECTED
    error_message: Optional[str] = None
    processed_update_ids: List[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TelegramThreadDocument(BaseModel):
    org_id: Optional[str] = None
    telegram_chat_id: int
    chat_type: str = "private"
    title: Optional[str] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    auto_reply: Optional[bool] = None
    unread_count: int = 0
    last_message_at: Optional[datetime] = None
    last_message_preview: Optional[str] = None
    agent_history: List[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TelegramMessageDocument(BaseModel):
    org_id: Optional[str] = None
    thread_id: str
    direction: TelegramDirection
    sender: TelegramSender
    text: str
    telegram_message_id: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
