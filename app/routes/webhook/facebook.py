import logging
from fastapi import APIRouter, Request, Query, Header, HTTPException, Depends, Response
from app.services.webhook.facebook_service import FacebookService
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook/facebook", tags=["webhook"])

from app.dependencies import get_facebook_service

@router.get("/")
def verify(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
    facebook_service: FacebookService = Depends(get_facebook_service)
):
    """
    Handle verification from Facebook.
    """
    logger.info(f"Incoming Facebook webhook verification challenge: mode={mode}, token={token}")
    challenge_response = facebook_service.verify_webhook(mode, token, challenge)
    logger.info("Facebook webhook verified successfully")
    return Response(content=challenge_response, media_type="text/plain")

@router.post("/")
async def handle_event(
    request: Request, 
    x_hub_signature_256: Optional[str] = Header(None),
    facebook_service: FacebookService = Depends(get_facebook_service)
):
    """
    Handle actual webhook events from Facebook.
    """
    logger.info("Received Facebook webhook event POST request")
    # Verify signature if app secret is configured
    if facebook_service.app_secret:
        body = await request.body()
        if not facebook_service.verify_signature(body, x_hub_signature_256):
            logger.warning("Facebook webhook event verification failed: invalid signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
            
    data = await request.json()
    logger.info(f"Processing Facebook webhook event payload: {data}")
    await facebook_service.handle_webhook_event(data)
    return {"status": "ok"}
