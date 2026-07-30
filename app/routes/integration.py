import logging
from typing import List

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_telegram_service
from app.models.telegram import (
    TelegramConnectRequest,
    TelegramIntegrationResponse,
    TelegramIntegrationUpdate,
    TelegramMessageResponse,
    TelegramSendRequest,
    TelegramThreadResponse,
    TelegramThreadUpdate,
)
from app.security import require_org_membership
from app.services.webhook.telegram_service import TelegramService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations/telegram", tags=["integrations"])


@router.get("/", response_model=TelegramIntegrationResponse)
async def get_telegram_integration(
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    return telegram_service.get_integration(org_id)


@router.post("/connect", response_model=TelegramIntegrationResponse)
async def connect_telegram(
    req: TelegramConnectRequest,
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    logger.info(f"Connecting a Telegram bot for org {org_id}")
    return await telegram_service.connect(org_id, req)


@router.patch("/", response_model=TelegramIntegrationResponse)
async def update_telegram_integration(
    req: TelegramIntegrationUpdate,
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    return telegram_service.update_integration(org_id, req)


@router.delete("/")
async def disconnect_telegram(
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    logger.info(f"Disconnecting the Telegram integration for org {org_id}")
    await telegram_service.disconnect(org_id)
    return {"status": "disconnected"}


@router.get("/threads", response_model=List[TelegramThreadResponse])
async def list_threads(
    limit: int = Query(50, ge=1, le=200),
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    return telegram_service.list_threads(org_id, limit=limit)


@router.get("/threads/{thread_id}", response_model=TelegramThreadResponse)
async def get_thread(
    thread_id: str,
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    return telegram_service.get_thread(org_id, thread_id)


@router.patch("/threads/{thread_id}", response_model=TelegramThreadResponse)
async def update_thread(
    thread_id: str,
    req: TelegramThreadUpdate,
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    return telegram_service.update_thread(org_id, thread_id, req)


@router.post("/threads/{thread_id}/read", response_model=TelegramThreadResponse)
async def mark_thread_read(
    thread_id: str,
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    return telegram_service.mark_read(org_id, thread_id)


@router.get(
    "/threads/{thread_id}/messages", response_model=List[TelegramMessageResponse]
)
async def list_thread_messages(
    thread_id: str,
    limit: int = Query(200, ge=1, le=500),
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    return telegram_service.list_messages(org_id, thread_id, limit=limit)


@router.post("/threads/{thread_id}/messages", response_model=TelegramMessageResponse)
async def send_thread_message(
    thread_id: str,
    req: TelegramSendRequest,
    org_id: str = Depends(require_org_membership),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    logger.info(
        f"Sending a manual Telegram reply on thread {thread_id} for org {org_id}"
    )
    return await telegram_service.send_manual_reply(org_id, thread_id, req.text)
