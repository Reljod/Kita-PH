from fastapi import APIRouter, Request, Query, Header, HTTPException, Depends
from app.services.webhook.facebook_service import FacebookService
from typing import Optional

router = APIRouter(prefix="/webhook/facebook", tags=["webhook"])

def get_facebook_service() -> FacebookService:
    return FacebookService()

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
    return facebook_service.verify_webhook(mode, token, challenge)

@router.post("/")
async def handle_event(
    request: Request, 
    x_hub_signature_256: Optional[str] = Header(None),
    facebook_service: FacebookService = Depends(get_facebook_service)
):
    """
    Handle actual webhook events from Facebook.
    """
    # Verify signature if app secret is configured
    if facebook_service.app_secret:
        body = await request.body()
        if not facebook_service.verify_signature(body, x_hub_signature_256):
            raise HTTPException(status_code=403, detail="Invalid signature")
            
    data = await request.json()
    print("Data: ", data)
    await facebook_service.handle_webhook_event(data)
    return {"status": "ok"}
