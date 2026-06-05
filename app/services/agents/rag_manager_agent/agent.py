import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.services.tools.file_tools import file_toolset
from app.services.tools.parse_tools import parse_toolset
from app.services.tools.agent_tools import agent_toolset
from app.services.tools.graph_rag_tools import graph_rag_toolset
from app.services.tools.delegation_tools import delegation_toolset


class RagManagerAgent(Agent):
    def __init__(self):
        model_name = os.getenv("LLM_MODEL", "x-ai/grok-4.3") # Use a powerful model for coordination
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
            retries=3,
            toolsets=[
                file_toolset, 
                parse_toolset, 
                agent_toolset, 
                graph_rag_toolset, 
                delegation_toolset
            ]
        )
