import os
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

def get_agent() -> Agent:
    model_name = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct") # OpenRouter model name
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    model = OpenRouterModel(
        model_name,
        provider=OpenRouterProvider(api_key=api_key),
    )
    
    agent = Agent(
        model=model,
        system_prompt='You are a helpful and concise AI assistant.'
    )
    return agent
