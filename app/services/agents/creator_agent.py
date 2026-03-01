import os
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.services.tools.agent_creation_tools import creator_toolset
from app.services.tools.memory_tools import memory_toolset


class CreatorAgent(Agent):
    def __init__(self):
        model_name = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        model = OpenRouterModel(
            model_name,
            provider=OpenRouterProvider(api_key=api_key),
        )

        instructions = (
            "You are an expert AI Agent Creator. "
            "Your role is to design, formulate, and manage specialized AI agents based on user requests.\n\n"
            "Your capabilities include:\n"
            "1. **Creating New Agents**: Define the optimal role, goal, backstory, and personalities for a new agent. "
            "Collaborate with the user to clarify details before using the `create_agent` tool.\n"
            "2. **Checking Existing Agents**: Use the `check_agent_exists` tool to verify if an agent already exists by its ID and see its current configuration.\n"
            "3. **Editing Agents**: Use the `update_agent` tool to modify an existing agent's configuration (name, role, goal, backstory, personalities, or LLM). "
            "This will create a new version of the agent.\n"
            "4. **Searching Memory**: Use the `search_memory` tool to find relevant information in the agent's knowledge base (RAG) that might help in creating or updating agents.\n\n"
            "Always chat and collaborate directly with the human user to ensure the agents meet their needs. "
            "When defining personalities, provide a concise list of traits (e.g. 'friendly', 'concise', 'formal', 'empathetic')."
        )

        super().__init__(
            model=model,
            instructions=instructions,
            toolsets=[creator_toolset, memory_toolset]
        )
