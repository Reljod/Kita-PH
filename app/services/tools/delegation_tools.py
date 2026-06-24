from pydantic_ai import FunctionToolset, RunContext
from pydantic import Field
from typing import Optional, Annotated
from app.utils.logger import log_tool_call

delegation_toolset = FunctionToolset()

@delegation_toolset.tool
@log_tool_call
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
    from app.services.guardrail_service import DEFAULT_GUARDRAILS
    
    agent_service: IAgentService = ctx.deps.get("agent_service")
    if not agent_service:
        return "Error: Agent service not found in dependencies."
    
    guardrails = ctx.deps.get("guardrails", DEFAULT_GUARDRAILS)
    max_depth = guardrails.get("max_delegation_depth", DEFAULT_GUARDRAILS["max_delegation_depth"])
    
    current_depth = ctx.deps.get("_delegation_depth", 0)
    if current_depth >= max_depth:
        return f"Error: Maximum delegation depth ({max_depth}) reached. Cannot delegate further."
    
    # Use target_agent_id if provided, otherwise fallback to current agent_id from deps
    actual_agent_id = target_agent_id or ctx.deps.get("agent_id")
    
    try:
        sub_agent = agent_service.get_runnable_agent(agent_id=actual_agent_id)
        
        # Track status if status_key is provided
        status_key = ctx.deps.get("status_key")
        if status_key:
            try:
                from app.dependencies.services import get_services
                services = get_services(ctx.deps.get("org_id"))
                await services.agent_status_service.update_step(status_key, "delegated_task", actual_agent_id)
            except Exception as e:
                import logfire
                logfire.error("Failed to update status in delegate_task: {error}", error=str(e))

        # Pass incremented depth to sub-agent via a shallow copy of deps
        child_deps = dict(ctx.deps)
        child_deps["_delegation_depth"] = current_depth + 1

        result = await sub_agent.run(
            task_description, 
            deps=child_deps,
            usage=ctx.usage
        )
        
        # Attempt to parse as JSON if it's a string, to return structured data to the parent
        output = result.output
        if isinstance(output, str):
            import json
            import re
            import logfire
            
            # Remove potential markdown code blocks
            clean_output = re.sub(r'```json\s*(.*?)\s*```', r'\1', output, flags=re.DOTALL).strip()
            
            try:
                parsed = json.loads(clean_output)
                
                # Diagnostic: Log the size and complexity
                logfire.info(
                    "Sub-agent returned structured JSON from task: {task}", 
                    task=task_description[:50],
                    keys=list(parsed.keys()) if isinstance(parsed, dict) else "list",
                    size_chars=len(clean_output)
                )
                
                # Safety: If the output is massive, stay as a string to avoid overwhelming the parent LLM with structured state
                if len(clean_output) > 25000: # ~25KB threshold
                     logfire.warning("Sub-agent output too large for structured return, falling back to string.")
                     return output
                     
                return parsed
            except json.JSONDecodeError:
                return output
        
        return output
    except Exception as e:
        return f"Error during delegation to agent '{actual_agent_id}': {str(e)}"
