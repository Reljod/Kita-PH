import os
import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import redis.asyncio as aioredis
import logfire

EMOJI_MAP = {
    "condense_query": "🧠",
    "route_query": "🚦",
    "retrieve_facts_vector": "🔍",
    "retrieve_facts_web": "🌐",
    "grade_relevance": "📊",
    "grade_groundedness": "⚖️",
    "grade_completeness": "🏁",
    "rewrite_query": "✍️",
    "generate_response": "🤖",
    "draft_response": "📝",
    "finalize_response": "🤖",
    "delegated_task": "🤝"
}

MESSAGES_MAP = {
    "condense_query": [
        "analyzing conversation history",
        "summarizing chat context",
        "gathering previous message history"
    ],
    "route_query": [
        "analyzing query intent",
        "determining best response strategy",
        "routing user request",
        "checking query intent"
    ],
    "retrieve_facts_vector": [
        "searching the internal knowledge base",
        "gathering facts from company documents",
        "fetching relevant context"
    ],
    "retrieve_facts_web": [
        "searching the web for fresh information",
        "querying external search engines",
        "performing web search"
    ],
    "grade_relevance": [
        "evaluating retrieved document relevance",
        "grading source relevance",
        "filtering out unrelated sources"
    ],
    "grade_groundedness": [
        "checking answer groundedness",
        "verifying draft logic against facts",
        "checking for hallucinations"
    ],
    "grade_completeness": [
        "evaluating response completeness",
        "checking if all details are answered",
        "verifying completeness"
    ],
    "rewrite_query": [
        "optimizing search queries",
        "rewriting query to improve results",
        "re-formulating search query"
    ],
    "generate_response": [
        "drafting a response",
        "generating the final answer",
        "formulating the reply"
    ],
    "draft_response": [
        "drafting a response",
        "formulating the response draft"
    ],
    "finalize_response": [
        "finalizing the answer",
        "writing the final reply"
    ],
    "delegated_task": [
        "delegating task to sub-agent",
        "spawning a specialized sub-agent",
        "awaiting sub-agent completion"
    ]
}


class AgentStepStatus(BaseModel):
    step: str
    message: str
    agent_id: str
    started_at: str
    completed_at: Optional[str] = None


class AgentStatus(BaseModel):
    status_key: str
    chat_id: Optional[str] = None
    agent_id: str
    active_agent: str
    status: str  # in_progress, completed, failed, awaiting_input
    current_step: Optional[str] = None
    current_message: Optional[str] = None
    steps: List[AgentStepStatus] = Field(default_factory=list)
    action_required: Optional[Dict[str, Any]] = None
    started_at: str
    updated_at: str


class AgentStatusService:
    def __init__(self, org_id: str, redis_client: aioredis.Redis, agent_service: Any = None):
        self.org_id = org_id
        self.redis = redis_client
        self.agent_service = agent_service

    def set_agent_service(self, agent_service: Any):
        self.agent_service = agent_service

    def _get_redis_key(self, status_key: str) -> str:
        return f"agent_status:{status_key}"

    def _get_channel_name(self, status_key: str) -> str:
        return f"agent_status:channel:{status_key}"

    def _get_agent_name(self, agent_id: str) -> str:
        if not agent_id or agent_id == "KitaAgent":
            return "KitaAgent"
        if self.agent_service:
            try:
                agent = self.agent_service.get_agent(agent_id)
                if agent:
                    return agent.name
            except Exception as e:
                logfire.warning("Failed to lookup agent name: {error}", error=str(e))
        return agent_id

    def _generate_step_message(self, step: str, agent_id: str) -> str:
        emoji = EMOJI_MAP.get(step, "🤖")
        agent_name = self._get_agent_name(agent_id)
        
        templates = MESSAGES_MAP.get(step, ["processing"])
        chosen_template = random.choice(templates)
        
        return f"{emoji} *{agent_name} is {chosen_template}*"

    async def get_status(self, status_key: str) -> Optional[Dict[str, Any]]:
        key = self._get_redis_key(status_key)
        data = await self.redis.get(key)
        if not data:
            return None
        try:
            return json.loads(data)
        except Exception:
            return None

    async def start_session(self, status_key: str, agent_id: str, chat_id: Optional[str] = None) -> AgentStatus:
        now_str = datetime.now(timezone.utc).isoformat()
        status = AgentStatus(
            status_key=status_key,
            chat_id=chat_id,
            agent_id=agent_id,
            active_agent=agent_id,
            status="in_progress",
            started_at=now_str,
            updated_at=now_str
        )
        
        await self._save_and_publish(status, ttl=3600)
        logfire.info("Started agent status session: key={key}", key=status_key)
        return status

    async def update_step(self, status_key: str, step: str, agent_id: Optional[str] = None) -> Optional[AgentStatus]:
        status_dict = await self.get_status(status_key)
        if not status_dict:
            return None
            
        status = AgentStatus.model_validate(status_dict)
        now_str = datetime.now(timezone.utc).isoformat()
        
        # Determine the agent running this step
        active_agent = agent_id or status.agent_id
        
        # Complete any existing active steps
        for s in status.steps:
            if not s.completed_at:
                s.completed_at = now_str
                
        # Generate new step message
        message = self._generate_step_message(step, active_agent)
        
        # Add new step
        new_step = AgentStepStatus(
            step=step,
            message=message,
            agent_id=active_agent,
            started_at=now_str
        )
        status.steps.append(new_step)
        
        status.current_step = step
        status.current_message = message
        status.active_agent = active_agent
        status.updated_at = now_str
        
        await self._save_and_publish(status, ttl=3600)
        logfire.info("Updated status step: key={key}, step={step}", key=status_key, step=step)
        return status

    async def finish_session(self, status_key: str, success: bool = True, chat_id: Optional[str] = None) -> Optional[AgentStatus]:
        status_dict = await self.get_status(status_key)
        if not status_dict:
            return None
            
        status = AgentStatus.model_validate(status_dict)
        now_str = datetime.now(timezone.utc).isoformat()
        
        # Complete any remaining open steps
        for s in status.steps:
            if not s.completed_at:
                s.completed_at = now_str
                
        status.status = "completed" if success else "failed"
        if chat_id:
            status.chat_id = chat_id
            
        status.current_step = None
        status.current_message = None
        status.updated_at = now_str
        
        # Set 5-minute TTL so frontend has time to receive final state
        await self._save_and_publish(status, ttl=300)
        logfire.info("Finished status session: key={key}, success={success}", key=status_key, success=success)
        return status

    async def _save_and_publish(self, status: AgentStatus, ttl: int):
        status_json = status.model_dump_json()
        key = self._get_redis_key(status.status_key)
        channel = self._get_channel_name(status.status_key)
        
        # Save to Redis
        await self.redis.set(key, status_json, ex=ttl)
        
        # Publish to PubSub channel
        await self.redis.publish(channel, status_json)
