from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, status
import os
from typing import List
from app.models.agent import (
    AgentCreateRequest, AgentUpdateRequest, AgentResponse,
    AddToolsRequest, RemoveToolsRequest
)
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest
from app.models.rag import RagCreateRequest, RagResponse, RagUpdateRequest
from app.services.agent_service import IAgentService
from app.services.chat_service import IChatService
from app.services.rag_service import IRagService
from app.security import get_current_org_id
from app.dependencies import get_agent_service, get_chat_service, get_agent_rag_service

router = APIRouter(prefix="/agent", tags=["Agent Management"])

@router.post("/", response_model=AgentResponse)
async def create_agent(
    req: AgentCreateRequest, 
    service: IAgentService = Depends(get_agent_service)
):
    try:
        return await service.create_agent(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str, 
    req: AgentUpdateRequest, 
    service: IAgentService = Depends(get_agent_service),
    new_version: bool = True
):
    agent = await service.update_agent(agent_id, req, new_version=new_version)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.get("/{agent_id}", response_model=AgentResponse)
def get_agent(agent_id: str, service: IAgentService = Depends(get_agent_service)):
    agent = service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.get("/", response_model=List[AgentResponse])
def get_all_agents(last_chat: bool = False, service: IAgentService = Depends(get_agent_service)):
    return service.get_all_agents(include_last_chat=last_chat)

@router.delete("/{agent_id}")
def delete_agent(agent_id: str, service: IAgentService = Depends(get_agent_service)):
    success = service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}

# (ChatService dependency is imported from app.dependencies)

@router.post("/{agent_id}/chat", response_model=ChatResponse)
async def create_agent_chat(agent_id: str, req: ChatCreateRequest, chat_service: IChatService = Depends(get_chat_service)):
    try:
        return await chat_service.create_chat(req, agent_id=agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")

@router.post("/{agent_id}/chat/{chat_id}/continue", response_model=ChatResponse)
async def continue_agent_chat(agent_id: str, chat_id: str, req: ChatContinueRequest, chat_service: IChatService = Depends(get_chat_service)):
    try:
        chat = await chat_service.continue_chat(chat_id, req, agent_id=agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")
        
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return chat

@router.get("/{agent_id}/chat/{chat_id}", response_model=ChatResponse)
async def get_agent_chat(agent_id: str, chat_id: str, chat_service: IChatService = Depends(get_chat_service)):
    try:
        chat = chat_service.get_chat(chat_id, agent_id=agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return chat

@router.get("/{agent_id}/chat", response_model=List[ChatResponse])
async def get_all_agent_chats(agent_id: str, preview: bool = False, chat_service: IChatService = Depends(get_chat_service)):
    return chat_service.get_all_chats(agent_id=agent_id, preview=preview)

# Agent Memory Routes

# (Agent RAG Service dependency is imported from app.dependencies)

@router.post("/{agent_id}/memory", response_model=RagResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_rag(
    agent_id: str,
    req: RagCreateRequest, 
    background_tasks: BackgroundTasks,
    rag_service: IRagService = Depends(get_agent_rag_service)
):
    try:
        # Ensure agent_id is set in the request if not already
        req.agent_id = agent_id
        rag = await rag_service.add_rag(req)
        # Trigger background embedding
        background_tasks.add_task(rag_service.update_embedding, rag.id)
        return rag
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent memory: {str(e)}")

@router.get("/{agent_id}/memory", response_model=List[RagResponse])
async def get_all_agent_rags(agent_id: str, rag_service: IRagService = Depends(get_agent_rag_service)):
    return rag_service.get_all_rags(agent_id=agent_id)

@router.get("/{agent_id}/memory/search", response_model=List[RagResponse])
async def search_agent_memory(
    agent_id: str,
    query: str = Query(..., description="The search query to find relevant information in memory."), 
    limit: int = Query(5, description="The maximum number of results to return."), 
    rag_service: IRagService = Depends(get_agent_rag_service)
):
    try:
        return await rag_service.search(query, limit=limit, agent_id=agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching agent memory: {str(e)}")

@router.get("/{agent_id}/memory/{rag_id}", response_model=RagResponse)
async def get_agent_rag(agent_id: str, rag_id: str, rag_service: IRagService = Depends(get_agent_rag_service)):
    try:
        rag = rag_service.get_rag(rag_id, agent_id=agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not rag:
        raise HTTPException(status_code=404, detail="Memory not found")
        
    return rag

@router.put("/{agent_id}/memory/{rag_id}", response_model=RagResponse)
async def update_agent_rag(
    agent_id: str,
    rag_id: str, 
    req: RagUpdateRequest, 
    background_tasks: BackgroundTasks,
    rag_service: IRagService = Depends(get_agent_rag_service)
):
    try:
        rag = await rag_service.edit_rag(rag_id, req, agent_id=agent_id)
        if not rag:
            raise HTTPException(status_code=404, detail="Memory not found")
            
        # If status is pending (which service sets if content updated), trigger background re-embedding
        if rag.status == "pending":
            background_tasks.add_task(rag_service.update_embedding, rag.id)
            
        return rag
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating agent memory: {str(e)}")

@router.delete("/{agent_id}/memory/{rag_id}")
async def delete_agent_rag(agent_id: str, rag_id: str, rag_service: IRagService = Depends(get_agent_rag_service)):
    try:
        success = await rag_service.delete_rag(rag_id, agent_id=agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"message": "Memory deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting agent memory: {str(e)}")

# Agent Tool Routes

@router.post("/{agent_id}/tools/add", response_model=AgentResponse)
async def add_agent_tools(
    agent_id: str,
    req: AddToolsRequest,
    service: IAgentService = Depends(get_agent_service)
):
    success = await service.add_tools(agent_id, req.tool_ids)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Return updated agent
    agent = service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.post("/{agent_id}/tools/remove", response_model=AgentResponse)
async def remove_agent_tools(
    agent_id: str,
    req: RemoveToolsRequest,
    service: IAgentService = Depends(get_agent_service)
):
    success = await service.remove_tools(agent_id, req.tool_ids)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Return updated agent
    agent = service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
