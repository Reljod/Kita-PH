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
        llm_id=os.getenv("LLM_MODEL", "x-ai/grok-4.3"),
        created_at=datetime(2026, 2, 25),
        updated_at=datetime(2026, 2, 25)
    ),
    "kita-assistant": AgentResponse(
        id="kita-assistant",
        base_id="kita-assistant",
        version=1,
        name="Kita Assistant",
        role="Helpful AI Assistant",
        goal="Provide concise and expert assistance with any task or question.",
        backstory="You are a state-of-the-art AI assistant, designed to be professional, helpful, and efficient. You have access to tools that enhance your capabilities.",
        personalities=["Helpful", "Professional", "Concise"],
        status="completed",
        llm_id=os.getenv("LLM_MODEL", "x-ai/grok-4.3"),
        created_at=datetime(2026, 3, 7),
        updated_at=datetime(2026, 3, 7)
    ),
    "rag-manager": AgentResponse(
        id="rag-manager",
        base_id="rag-manager",
        version=1,
        name="Rag Manager",
        role="Expert Rag Manager Agent",
        goal="Orchestrate the ingestion of documents into a Meta-Ontology Graph RAG system.",
        backstory="You are the expert Rag Manager Agent. Your role is to orchestrate the ingestion of documents into a Meta-Ontology Graph RAG system. You handle file resolution, parse retrieval, sliding window chunking, and delegation to specialized agents.",
        status="completed",
        llm_id=os.getenv("LLM_MODEL", "x-ai/grok-4.3"),
        created_at=datetime(2026, 4, 7),
        updated_at=datetime(2026, 4, 7)
    )
}

BASE_AGENT = SYSTEM_AGENTS["kita-assistant"]
