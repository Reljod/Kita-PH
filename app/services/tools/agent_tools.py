from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Annotated, Optional, List
from app.utils.logger import log_tool_call

agent_toolset = FunctionToolset()

@agent_toolset.tool
@log_tool_call
async def get_available_agents(
    ctx: RunContext[dict]
) -> List[dict]:
    """
    Returns a list of all available specialized agents in the organization.
    Each agent includes its 'id', 'name', 'role', and 'goal'.
    Use this to identify which agent is best suited for a specific sub-task.
    """
    from app.services.agent_service import IAgentService
    
    agent_service: IAgentService = ctx.deps.get("agent_service")
    if not agent_service:
        return [{"error": "Agent service not found in dependencies."}]
    
    current_agent_id = ctx.deps.get("agent_id")
    
    try:
        all_agents = agent_service.get_all_agents()
        available_agents = []
        
        for agent in all_agents:
            # Skip the current agent to avoid infinite delegation loops
            if agent.id == current_agent_id:
                continue
                
            available_agents.append({
                "id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "goal": agent.goal
            })
        
        return available_agents
    except Exception as e:
        return [{"error": f"Error fetching available agents: {str(e)}"}]
