from fastapi import APIRouter, HTTPException, Depends, Header
import os
from typing import List, Optional
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest
from app.services.chat_service import IChatService
from app.dependencies import get_chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def create_chat(
    req: ChatCreateRequest, 
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
):
    try:
        return await chat_service.create_chat(req, agent_id=x_agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")

@router.post("/{chat_id}/continue", response_model=ChatResponse)
async def continue_chat(
    chat_id: str, 
    req: ChatContinueRequest, 
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
):
    try:
        chat = await chat_service.continue_chat(chat_id, req, agent_id=x_agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")
        
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return chat

@router.get("/{chat_id}", response_model=ChatResponse)
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

@router.get("", response_model=List[ChatResponse])
async def get_all_chats(
    chat_service: IChatService = Depends(get_chat_service),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
):
    return chat_service.get_all_chats(agent_id=x_agent_id)
