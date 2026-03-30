from pydantic_ai import FunctionToolset
from pydantic import Field
from typing import List, Optional, Annotated
from pydantic_ai import RunContext
from app.db import db, TenantCollection
from app.models.agent import AgentDocument, format_agent_response

creator_toolset = FunctionToolset()

@creator_toolset.tool
async def create_agent(
    ctx: RunContext[dict],
    name: Annotated[str, Field(description="The name of the new agent.")],
    role: Annotated[str, Field(description="The role assigned to the new agent.")],
    goal: Annotated[str, Field(description="The goal of the new agent.")],
    backstory: Annotated[str, Field(description="The backstory of the new agent.")],
    personalities: Annotated[Optional[List[str]], Field(description="A list of personality traits for the agent (e.g. 'friendly', 'concise', 'formal', 'empathetic'). These shape how the agent communicates.")] = None,
    llm_id: Annotated[Optional[str], Field(description="The ID of the LLM model to use for this agent. If not provided, it will be automatically selected based on environment defaults.")] = None
) -> str:
    """
    Finalizes the details of a new agent and saves it to the database.
    Call this tool when you have collected all the necessary information from the user 
    about the new agent they want to create.
    """
    import os
    from app.services.llm_service import LlmService
    from app.services.agent_service import AgentService
    from app.models.agent import AgentCreateRequest

    org_id = ctx.deps["org_id"]
    llm_service_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_service_coll)
    
    if not llm_id or llm_id == "null":
        available_llms = llm_service.list_llms()
        env_model = os.getenv("LLM_MODEL")
        
        # Try to match with LLM_MODEL from .env
        match = next((l for l in available_llms if l.model == env_model), None)
        if not match:
            # Try to match with grok-4.1-fast fallback
            match = next((l for l in available_llms if l.model == "x-ai/grok-4.1-fast"), None)
            
        llm_id = match.id if match else ""

    agent_coll = TenantCollection(db.get_agents_collection(), org_id)
    
    agent_service = AgentService(
        llm_service=llm_service,
        collection=agent_coll
    )

    req = AgentCreateRequest(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        personalities=personalities,
        llm_id=llm_id
    )
    agent = await agent_service.create_agent(req)
    
    return f"Successfully created agent '{name}' with ID: {agent.id}"

@creator_toolset.tool
async def get_agent(
    ctx: RunContext[dict],
    agent_id: Annotated[str, Field(description="The ID of the agent to get (e.g., '67bc...-v1' or '67bc...')")]
) -> str:
    """
    Retrieves full information about an agent by its ID, including the LLM model used.
    """
    from app.models.agent import parse_agent_id
    from app.services.llm_service import LlmService

    base_id, version = parse_agent_id(agent_id)
    
    query = {"base_id": base_id}
    if version:
        query["version"] = version
        
    org_id = ctx.deps["org_id"]
    collection = TenantCollection(db.get_agents_collection(), org_id)
    doc = collection.find_one(query)
    if not doc:
        return f"Agent with ID '{agent_id}' does not exist."
    
    agent = format_agent_response(doc)
    
    # Get LLM info
    llm_service_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_service_coll)
    llm = llm_service.get_llm(agent.llm_id)
    
    llm_info = f"{llm.name} ({llm.model} via {llm.provider})" if llm else f"Unknown LLM (ID: {agent.llm_id})"

    return (
        f"Agent: {agent.name}\n"
        f"ID: {agent.id}\n"
        f"Base ID: {agent.base_id}\n"
        f"LLM: {llm_info}\n"
        f"Role: {agent.role}\n"
        f"Goal: {agent.goal}\n"
        f"Backstory: {agent.backstory}\n"
        f"Personalities: {', '.join(agent.personalities) if agent.personalities else 'None'}\n"
        f"Tools: {', '.join(agent.tools) if agent.tools else 'None'}"
    )

@creator_toolset.tool
async def list_agents(
    ctx: RunContext[dict]
) -> str:
    """
    Retrieves a list of all existing agents in the organization.
    """
    from app.services.agent_service import AgentService
    from app.services.llm_service import LlmService

    org_id = ctx.deps["org_id"]
    llm_service_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_service_coll)
    agent_coll = TenantCollection(db.get_agents_collection(), org_id)
    
    service = AgentService(llm_service=llm_service, collection=agent_coll)
    agents = service.get_all_agents()
    
    if not agents:
        return "No agents found."
    
    result = "Existing Agents:\n"
    for agent in agents:
        result += f"- {agent.name} (ID: {agent.id}, Base ID: {agent.base_id})\n"
    
    return result

@creator_toolset.tool
async def update_agent(
    ctx: RunContext[dict],
    agent_id: Annotated[str, Field(description="The ID of the agent to update.")],
    name: Annotated[Optional[str], Field(description="The new name of the agent.")] = None,
    role: Annotated[Optional[str], Field(description="The new role of the agent.")] = None,
    goal: Annotated[Optional[str], Field(description="The new goal of the agent.")] = None,
    backstory: Annotated[Optional[str], Field(description="The new backstory of the agent.")] = None,
    personalities: Annotated[Optional[List[str]], Field(description="Updated list of personality traits for the agent.")] = None,
    llm_id: Annotated[Optional[str], Field(description="The new LLM ID for the agent.")] = None,
    new_version: Annotated[bool, Field(description="Whether to create a new version (True) or update the current version in place (False). Set to False for iterating/tweaking.")] = True
) -> str:
    """
    Updates an existing agent's configuration.
    By default, this creates a new version. Set new_version=False to update the current version in place.
    """
    from app.models.agent import parse_agent_id, AgentUpdateRequest
    from app.services.llm_service import LlmService
    from app.services.agent_service import AgentService
    
    org_id = ctx.deps["org_id"]
    llm_service_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_service_coll)
    agent_coll = TenantCollection(db.get_agents_collection(), org_id)
    service = AgentService(llm_service=llm_service, collection=agent_coll)
    
    update_req = AgentUpdateRequest(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        personalities=personalities,
        llm_id=llm_id,
    )
    
    updated_agent = await service.update_agent(agent_id, update_req, new_version=new_version)
    if not updated_agent:
        return f"Failed to update agent '{agent_id}'. It might not exist."
        
    return f"Successfully updated agent: {updated_agent.id}"
