from pydantic_ai import FunctionToolset
from pydantic_ai import RunContext
from app.db import db, TenantCollection
from app.services.llm_service import LlmService
from app.utils.logger import log_tool_call

llm_toolset = FunctionToolset()

@llm_toolset.tool
@log_tool_call
async def list_available_llms(
    ctx: RunContext[dict]
) -> str:
    """
    Lists all available Large Language Models (LLMs) for the organization.
    Use this to show the user which models they can choose from when creating or updating an agent.
    """
    org_id = ctx.deps["org_id"]
    llm_service_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_service_coll)
    
    available_llms = llm_service.list_llms()
    
    if not available_llms:
        return "No LLMs found for this organization."
    
    response = "Available LLMs:\n"
    for llm in available_llms:
        response += f"- {llm.name} (Model: {llm.model}, Provider: {llm.provider}, ID: {llm.id})\n"
    
    return response
