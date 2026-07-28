import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request

from app.dependencies import get_telegram_service
from app.services.webhook.telegram_service import TelegramService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook/telegram", tags=["webhook"])


@router.post("/{webhook_id}")
async def handle_update(
    webhook_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
    telegram_service: TelegramService = Depends(get_telegram_service),
):
    """Receive one update from Telegram.

    Answers as soon as the delivery is authenticated and hands the actual work
    to a background task: Telegram redelivers anything it is not acknowledged
    for quickly, and an agent run takes far longer than that window. The work
    is made idempotent by `update_id` deduplication rather than by holding the
    connection open.
    """
    integration = telegram_service.resolve_webhook(
        webhook_id, x_telegram_bot_api_secret_token
    )
    if not integration:
        raise HTTPException(status_code=403, detail="Invalid webhook credentials")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed update payload")

    background_tasks.add_task(telegram_service.process_update, integration, payload)
    return {"ok": True}
