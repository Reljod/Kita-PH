import os
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.services.tools.agent_creation_tools import creator_toolset
from app.services.tools.memory_tools import memory_toolset
from app.services.tools.llm_tools import llm_toolset
from app.services.tools.delegation_tools import delegation_toolset


class CreatorAgent(Agent):
    def __init__(self):
        model_name = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        model = OpenRouterModel(
            model_name,
            provider=OpenRouterProvider(api_key=api_key),
        )

        # Load instructions from instructions.md
        instructions_path = Path(__file__).parent / "instructions.md"
        instructions = instructions_path.read_text(encoding="utf-8")

        super().__init__(
            model=model,
            instructions=instructions,
            toolsets=[creator_toolset, memory_toolset, llm_toolset, delegation_toolset]
        )
