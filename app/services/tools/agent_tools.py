from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Annotated, Optional, List

agent_toolset = FunctionToolset()

@agent_toolset.tool
async def find_specialized_agent(
    ctx: RunContext[dict],
    document_type_hint: Annotated[str, Field(description="The type of document or content (e.g., 'financial report', 'technical manual')")]
) -> str:
    """
    Searches for a specialized agent in the organization that matches the content type.
    If no specialized agent is found, returns the ID of the current agent to handle the task normally.
    """
    from app.services.agent_service import IAgentService
    
    agent_service: IAgentService = ctx.deps.get("agent_service")
    if not agent_service:
        return "Error: Agent service not found in dependencies."
    
    current_agent_id = ctx.deps.get("agent_id", "rag-manager")
    
    try:
        all_agents = agent_service.get_all_agents()
        # Search for agents that are NOT system agents and match the hint
        # We can perform a simple string match on role, goal, or backstory
        hint_lower = document_type_hint.lower()
        
        for agent in all_agents:
            # Skip system agents (except if they are specifically relevant, but usually we want user-created ones)
            if agent.id.startswith("agent-") or agent.id == "kita-assistant":
                continue
                
            match_str = f"{agent.role} {agent.goal} {agent.backstory}".lower()
            if hint_lower in match_str:
                return agent.id
        
        return current_agent_id
    except Exception as e:
        return f"Error finding specialized agent: {str(e)}. Falling back to '{current_agent_id}'."
