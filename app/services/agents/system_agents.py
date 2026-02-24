import os
from datetime import datetime
from app.models.agent import AgentResponse

SYSTEM_AGENTS = {
    "agent-creator": AgentResponse(
        id="agent-creator",
        base_id="agent-creator",
        version=1,
        name="Agent Creator",
        role="Expert AI Agent Creator",
        goal="Design and formulate specialized AI agents based on user requests.",
        backstory="You are an expert AI Agent Creator. Your role is to design and formulate specialized AI agents based on user requests. You create agents of different roles, goals, and backstories.",
        system_prompt="Gather details about a new agent and use the creator tool.",
        status="completed",
        llm_id=os.getenv("LLM_MODEL", "x-ai/grok-4.1-fast"),
        created_at=datetime(2026, 2, 25),
        updated_at=datetime(2026, 2, 25)
    )
}
