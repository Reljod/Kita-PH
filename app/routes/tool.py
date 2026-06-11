from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.tool import ToolResponse, format_tool_response, ToolRegisterRequest
from app.models.agent import AgentResponse
from app.services.tool_service import ToolService, IToolService
from app.services.agent_service import AgentService, IAgentService
from app.services.llm_service import LlmService
from app.services.tools import get_available_tools
from app.services.web_search_service import SerperSearchService
from app.db import db, TenantCollection
from app.security import get_current_org_id

router = APIRouter(prefix="/tool", tags=["Tool Management"])

from app.dependencies import get_tool_service, get_agent_service

@router.get("/", response_model=List[ToolResponse])
async def get_tools(service: IToolService = Depends(get_tool_service)):
    try:
        tools = await service.get_tools()
        return [format_tool_response(t) for t in tools]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available")
async def get_all_available_tools():
    try:
        return get_available_tools()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", status_code=201)
async def register_tool(req: ToolRegisterRequest, service: IToolService = Depends(get_tool_service)):
    try:
        success = await service.register_tool(req.name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register tool")
        return {"message": "Tool registered successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: str, service: IToolService = Depends(get_tool_service)):
    try:
        tool = await service.get_tool(tool_id)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        return format_tool_response(tool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/name/{name}", response_model=ToolResponse)
async def get_tool_by_name(name: str, service: IToolService = Depends(get_tool_service)):
    try:
        tool = await service.get_tool_by_name(name)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        return format_tool_response(tool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{tool_id}")
async def deregister_tool(tool_id: str, service: IToolService = Depends(get_tool_service)):
    try:
        success = await service.deregister_tool(tool_id)
        if not success:
            raise HTTPException(status_code=404, detail="Tool not found")
        return {"message": f"Tool '{tool_id}' deregistered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{tool_id}/agents", response_model=List[AgentResponse])
async def get_tool_agents(
    tool_id: str, 
    agent_service: IAgentService = Depends(get_agent_service)
):
    try:
        return agent_service.get_agents_by_tool(tool_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
