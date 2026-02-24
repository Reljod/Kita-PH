from pydantic_ai import FunctionToolset
from pydantic import Field
from typing import Optional, Annotated
from app.db import db
from app.models.agent import AgentDocument, format_agent_response

creator_toolset = FunctionToolset()

@creator_toolset.tool
async def create_agent(
    name: Annotated[str, Field(description="The name of the new agent.")],
    role: Annotated[str, Field(description="The role assigned to the new agent.")],
    goal: Annotated[str, Field(description="The goal of the new agent.")],
    backstory: Annotated[str, Field(description="The backstory of the new agent.")],
    instructions: Annotated[str, Field(description="Instructions for the agent (this will be used as the system prompt).")],
    llm_id: Annotated[Optional[str], Field(description="The ID of the LLM model to use for this agent. If not provided, it will be automatically selected based on environment defaults.")] = None
) -> str:
    """
    Finalizes the details of a new agent and saves it to the database.
    Call this tool when you have collected all the necessary information from the user 
    about the new agent they want to create.
    """
    import os
    from app.services.llm_service import LlmService
    
    if not llm_id or llm_id == "null":
        llm_service = LlmService()
        available_llms = llm_service.list_llms()
        env_model = os.getenv("LLM_MODEL")
        
        # Try to match with LLM_MODEL from .env
        match = next((l for l in available_llms if l.model == env_model), None)
        if not match:
            # Try to match with grok-4.1-fast fallback
            match = next((l for l in available_llms if l.model == "x-ai/grok-4.1-fast"), None)
            
        llm_id = match.id if match else ""

    new_agent = AgentDocument(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        llm_id=llm_id,
        system_prompt=instructions,
        status="completed",
        version=1,
        base_id=None
    )
    
    doc = new_agent.model_dump()
    res = db.get_agents_collection().insert_one(doc)
    
    base_id_str = str(res.inserted_id)
    db.get_agents_collection().update_one(
        {"_id": res.inserted_id},
        {"$set": {"base_id": base_id_str}}
    )
    doc["_id"] = res.inserted_id
    doc["base_id"] = base_id_str
    
    response = format_agent_response(doc)
    return f"Successfully created agent '{name}' with ID: {response.id}"

@creator_toolset.tool
async def check_agent_exists(
    agent_id: Annotated[str, Field(description="The ID of the agent to check (e.g., '67bc...-v1' or '67bc...')")]
) -> str:
    """
    Checks if an agent exists in the database by its ID.
    """
    from app.models.agent import parse_agent_id
    base_id, version = parse_agent_id(agent_id)
    
    query = {"base_id": base_id}
    if version:
        query["version"] = version
        
    doc = db.get_agents_collection().find_one(query)
    if not doc:
        return f"Agent with ID '{agent_id}' does not exist."
    
    agent = format_agent_response(doc)
    return (
        f"Agent found: {agent.name}\n"
        f"Role: {agent.role}\n"
        f"Goal: {agent.goal}\n"
        f"Backstory: {agent.backstory}\n"
        f"Status: {agent.status}"
    )

@creator_toolset.tool
async def update_agent(
    agent_id: Annotated[str, Field(description="The ID of the agent to update.")],
    name: Annotated[Optional[str], Field(description="The new name of the agent.")] = None,
    role: Annotated[Optional[str], Field(description="The new role of the agent.")] = None,
    goal: Annotated[Optional[str], Field(description="The new goal of the agent.")] = None,
    backstory: Annotated[Optional[str], Field(description="The new backstory of the agent.")] = None,
    instructions: Annotated[Optional[str], Field(description="New system instructions for the agent.")] = None,
    llm_id: Annotated[Optional[str], Field(description="The new LLM ID for the agent.")] = None,
    new_version: Annotated[bool, Field(description="Whether to create a new version (True) or update the current version in place (False). Set to False for iterating/tweaking prompts.")] = True
) -> str:
    """
    Updates an existing agent's configuration.
    By default, this creates a new version. Set new_version=False to update the current version in place.
    """
    from app.models.agent import parse_agent_id, AgentUpdateRequest
    from app.services.llm_service import LlmService
    from app.services.agent_service import AgentService
    from app.services.agents.creator_agent import CreatorAgentService
    
    # We use AgentService to handle the versioning and update logic
    llm_service = LlmService()
    prompt_writer = CreatorAgentService()
    service = AgentService(llm_service=llm_service, prompt_writer_service=prompt_writer)
    
    update_req = AgentUpdateRequest(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        llm_id=llm_id,
        system_prompt=instructions
    )
    
    updated_agent = await service.update_agent(agent_id, update_req, new_version=new_version)
    if not updated_agent:
        return f"Failed to update agent '{agent_id}'. It might not exist."
        
    return f"Successfully updated agent: {updated_agent.id}"
