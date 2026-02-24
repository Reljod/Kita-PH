import os
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from typing import Protocol
from pydantic import BaseModel
from app.services.llm_service import LlmService

class PromptOutput(BaseModel):
    system_prompt: str

class IPromptWriterAgentService(Protocol):
    async def generate_prompt(self, role: str, goal: str, backstory: str, llm_id: str) -> str:
        ...

class PromptWriterAgentService(IPromptWriterAgentService):
    def __init__(self, llm_service: LlmService):
        self.llm_service = llm_service

    async def generate_prompt(self, role: str, goal: str, backstory: str, llm_id: str) -> str:
        llm = self.llm_service.get_llm(llm_id)
        if not llm:
            raise ValueError("LLM not found")

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model = OpenRouterModel(
            llm.model,
            provider=OpenRouterProvider(api_key=api_key),
        )

        agent = Agent(
            model=model,
            output_type=PromptOutput,
            system_prompt=(
                "You are an expert prompt engineer. Your job is to write a highly effective, "
                "clear, and structured System Prompt for an AI agent given its Role, Goal, and Backstory.\\n"
                "Return ONLY the system prompt text. Do not include introductory phrases or formatting like 'Here is the prompt:'."
            )
        )

        user_prompt = f"Role: {role}\\nGoal: {goal}\\nBackstory: {backstory}"
        result = await agent.run(user_prompt)
        if not result.output:
            raise ValueError("Failed to generate prompt")

        return result.output.system_prompt
