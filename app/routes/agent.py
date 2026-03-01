from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.agent import AgentCreateRequest, AgentUpdateRequest, AgentResponse
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest
from app.services.agent_service import AgentService, IAgentService
from app.services.chat_service import ChatService, IChatService
from app.services.llm_service import LlmService
from app.db import db, TenantCollection
from app.security import get_current_org_id

router = APIRouter(prefix="/agent", tags=["Agent Management"])

def get_agent_service(org_id: str = Depends(get_current_org_id)) -> IAgentService:
    llm_service_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_service_coll)
    collection = TenantCollection(db.get_agents_collection(), org_id)
    return AgentService(llm_service=llm_service, collection=collection)

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

def get_chat_service(
    org_id: str = Depends(get_current_org_id),
    agent_service: IAgentService = Depends(get_agent_service)
) -> IChatService:
    collection = TenantCollection(db.get_chats_collection(), org_id)
    return ChatService(agent_service, collection)

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
