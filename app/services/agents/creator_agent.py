from pydantic_ai import ToolOutput
import os
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.services.tools.agent_creation_tools import creator_toolset, create_agent, check_agent_exists, update_agent
from app.services.agents.prompt_writer_agent_service import IPromptWriterAgentService

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
            "1. **Creating New Agents**: Define the optimal role, goal, backstory, and system instructions for a new agent. "
            "Collaborate with the user to clarify details before using the `create_agent` tool.\n"
            "2. **Checking Existing Agents**: Use the `check_agent_exists` tool to verify if an agent already exists by its ID and see its current configuration.\n"
            "3. **Editing Agents**: Use the `update_agent` tool to modify an existing agent's configuration (name, role, goal, backstory, instructions, or LLM). "
            "This will create a new version of the agent.\n\n"
            "Always chat and collaborate directly with the human user to ensure the agents meet their needs. "
            "When generating system prompt instructions, be thorough and structured."
        )

        super().__init__(
            model=model,
            instructions=instructions,
            toolsets=[creator_toolset]
        )

class CreatorAgentService(IPromptWriterAgentService):
    async def generate_prompt(self, role: str, goal: str, backstory: str, llm_id: str) -> str:
        """
        Uses the CreatorAgent logic to generate a high-quality system prompt.
        """
        agent = CreatorAgent()
        user_prompt = (
            f"Please generate a comprehensive and structured system prompt for an agent with the following details:\n"
            f"Role: {role}\n"
            f"Goal: {goal}\n"
            f"Backstory: {backstory}\n\n"
            "Return ONLY the system prompt text. Do not include introductory phrases or formatting like 'Here is the prompt:'."
        )
        
        result = await agent.run(user_prompt)
        # The CreatorAgent doesn't have a structured output type like PromptWriterAgentService, 
        # so we take the raw data result.
        if not result.data:
            raise ValueError("Failed to generate prompt using CreatorAgent")
            
        return str(result.data)
