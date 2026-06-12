import json
from fastapi import APIRouter, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, Query
from typing import List, Optional
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest
from app.services.chat_service import IChatService
from app.dependencies import get_chat_service
from app.security import require_org_membership
from app.services.auth_service import AuthService

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse, dependencies=[Depends(require_org_membership)])
async def create_chat(
    req: ChatCreateRequest, 
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id"),
    x_status_key: Optional[str] = Header(None, alias="x-status-key")
):
    try:
        return await chat_service.create_chat(req, agent_id=x_agent_id, status_key=x_status_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")

@router.post("/{chat_id}/continue", response_model=ChatResponse, dependencies=[Depends(require_org_membership)])
async def continue_chat(
    chat_id: str, 
    req: ChatContinueRequest, 
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id"),
    x_status_key: Optional[str] = Header(None, alias="x-status-key")
):
    try:
        chat = await chat_service.continue_chat(chat_id, req, agent_id=x_agent_id, status_key=x_status_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")
        
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return chat

@router.get("/{chat_id}", response_model=ChatResponse, dependencies=[Depends(require_org_membership)])
async def get_chat(
    chat_id: str, 
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
):
    try:
        chat = chat_service.get_chat(chat_id, agent_id=x_agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return chat

@router.get("", response_model=List[ChatResponse], dependencies=[Depends(require_org_membership)])
async def get_all_chats(
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
):
    return chat_service.get_all_chats(agent_id=x_agent_id)

@router.get("/status/{status_key}", dependencies=[Depends(require_org_membership)])
async def get_agent_status(
    status_key: str,
    org_id: str = Depends(require_org_membership)
):
    from app.dependencies.services import get_services
    status_service = get_services(org_id).agent_status_service
    status = await status_service.get_status(status_key)
    if not status:
        raise HTTPException(status_code=404, detail="Status not found")
    return status

@router.websocket("/status/ws/{status_key}")
async def status_websocket(
    websocket: WebSocket,
    status_key: str,
    token: str = Query(...)
):
    # Authenticate manually via query parameter
    auth_service = AuthService()
    token_data = auth_service.verify_token(token)
    if not token_data or not token_data.org_id:
        await websocket.close(code=1008)
        return

    org_id = token_data.org_id
    await websocket.accept()

    from app.dependencies.services import get_services
    services = get_services(org_id)
    status_service = services.agent_status_service

    try:
        # Send initial status
        initial_status = await status_service.get_status(status_key)
        if initial_status:
            await websocket.send_json(initial_status)

        # Subscribe to PubSub
        pubsub = status_service.redis.pubsub()
        channel = status_service._get_channel_name(status_key)
        await pubsub.subscribe(channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                    if data.get("status") in ["completed", "failed"]:
                        break
        except WebSocketDisconnect:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    except Exception as e:
        import logfire
        logfire.error("WebSocket status trace failed: {error}", error=str(e))
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

