from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from app.models.agent import AgentCreateRequest, AgentUpdateRequest, AgentResponse
from app.services.agent_service import AgentService, IAgentService
from app.services.llm_service import LlmService
from app.services.agents.prompt_writer_agent_service import PromptWriterAgentService

router = APIRouter(prefix="/agent", tags=["Agent Management"])

def get_agent_service() -> IAgentService:
    llm_service = LlmService()
    prompt_writer = PromptWriterAgentService(llm_service=llm_service)
    return AgentService(llm_service=llm_service, prompt_writer_service=prompt_writer)

@router.post("/", response_model=AgentResponse)
async def create_agent(
    req: AgentCreateRequest, 
    background_tasks: BackgroundTasks,
    service: IAgentService = Depends(get_agent_service)
):
    try:
        agent = await service.create_agent(req)
        background_tasks.add_task(service.generate_prompt_background, agent.id)
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, req: AgentUpdateRequest, service: IAgentService = Depends(get_agent_service)):
    agent = await service.update_agent(agent_id, req)
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
def get_all_agents(service: IAgentService = Depends(get_agent_service)):
    return service.get_all_agents()

@router.delete("/{agent_id}")
def delete_agent(agent_id: str, service: IAgentService = Depends(get_agent_service)):
    success = service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}

@router.post("/{agent_id}/regenerate-prompt", response_model=AgentResponse)
async def regenerate_prompt(
    agent_id: str, 
    background_tasks: BackgroundTasks,
    service: IAgentService = Depends(get_agent_service)
):
    try:
        agent = await service.regenerate_prompt(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        background_tasks.add_task(service.generate_prompt_background, agent.id)
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
