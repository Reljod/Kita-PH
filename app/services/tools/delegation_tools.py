from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Optional, Annotated

delegation_toolset = FunctionToolset()

@delegation_toolset.tool
async def delegate_task(
    ctx: RunContext[dict],
    task_description: Annotated[str, Field(description="The precise task or research query to delegate to the sub-agent.")],
    target_agent_id: Annotated[Optional[str], Field(description="The ID of the agent to delegate to. If not provided, a sub-agent of the current agent will be used.")] = None
) -> str:
    """
    Spawns a sub-agent to handle a specific task, research item, or complex calculation.
    Use this to reduce the main conversation's context size or to perform multiple tasks in parallel.
    Multiple calls to this tool in a single turn will execute concurrently.
    
    Wait for the sub-agent to complete its task and return the results.
    """
    from app.services.agent_service import IAgentService
    
    agent_service: IAgentService = ctx.deps.get("agent_service")
    if not agent_service:
        return "Error: Agent service not found in dependencies."
    
    # Use target_agent_id if provided, otherwise fallback to current agent_id from deps
    actual_agent_id = target_agent_id or ctx.deps.get("agent_id")
    
    try:
        sub_agent = agent_service.get_runnable_agent(agent_id=actual_agent_id)
        # Run sub-agent with the provided task.
        # We pass the same deps structure to allow further nesting.
        # Passing ctx.usage allows pydantic-ai to aggregate token counts across agents.
        result = await sub_agent.run(
            task_description, 
            deps={
                "org_id": ctx.deps.get("org_id"), 
                "agent_id": actual_agent_id,
                "agent_service": agent_service
            },
            usage=ctx.usage
        )
        return str(result.output)
    except Exception as e:
        return f"Error during delegation to agent '{actual_agent_id}': {str(e)}"
