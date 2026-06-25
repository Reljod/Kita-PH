import json
import logging
from fastapi import APIRouter, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest
from app.services.chat_service import IChatService
from app.dependencies import get_chat_service
from app.security import require_org_membership
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", dependencies=[Depends(require_org_membership)])
async def create_chat(
    req: ChatCreateRequest, 
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id"),
    x_status_key: Optional[str] = Header(None, alias="x-status-key")
):
    logger.info("Initializing new chat stream")
    async def event_generator():
        try:
            async for event in chat_service.create_chat_stream(req, agent_id=x_agent_id, status_key=x_status_key):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            err_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/{chat_id}/continue", dependencies=[Depends(require_org_membership)])
async def continue_chat(
    chat_id: str, 
    req: ChatContinueRequest, 
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id"),
    x_status_key: Optional[str] = Header(None, alias="x-status-key")
):
    logger.info(f"Continuing chat stream for chat_id: {chat_id}")
    async def event_generator():
        try:
            async for event in chat_service.continue_chat_stream(chat_id, req, agent_id=x_agent_id, status_key=x_status_key):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Error in continue chat stream for chat_id {chat_id}: {e}", exc_info=True)
            err_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

@router.get("/{chat_id}/messages", dependencies=[Depends(require_org_membership)])
async def get_chat_messages(
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    chat_service: IChatService = Depends(get_chat_service),
    org_id: str = Depends(require_org_membership),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id"),
):
    chat = chat_service.get_chat(chat_id, agent_id=x_agent_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    from app.db import db

    coll = db.get_chat_archives_collection()
    cursor = (
        coll.find({"chat_id": chat_id, "org_id": org_id})
        .sort("created_at", -1)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    archives = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        archives.append(doc)

    return {
        "messages": chat.messages,
        "archives": archives,
        "page": page,
        "page_size": page_size,
    }

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
    user_id = token_data.user_id
    
    # WebSocket handshake is outside standard HTTP middleware context, set logging parameters manually
    from app.utils.logger import set_logging_context
    set_logging_context(
        org_id=org_id,
        user_id=user_id,
        request_id=f"ws-{status_key}",
        trace_id=f"ws-{status_key}"
    )

    logger.info(f"WebSocket status connection accepted for key: {status_key}")
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
            logger.info(f"WebSocket client disconnected for key: {status_key}")
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    except Exception as e:
        logger.error(f"WebSocket status trace failed for key {status_key}: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

