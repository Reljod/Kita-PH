import os
from bson import ObjectId
from typing import List, Optional, Protocol
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.models.agent import (
    AgentCreateRequest, AgentUpdateRequest, AgentResponse, AgentDocument,
    parse_agent_id, format_agent_response
)
from app.services.llm_service import ILlmService
from app.services.agents.creator_agent import IPromptWriterAgentService
from app.services.agents.system_agents import SYSTEM_AGENTS
from app.services.agents.system_prompt import get_full_instructions
from app.services.tools.memory_tools import memory_toolset
from app.db import TenantCollection

class IAgentService(Protocol):
    async def create_agent(self, req: AgentCreateRequest) -> AgentResponse: ...
    async def update_agent(self, agent_id: str, req: AgentUpdateRequest, new_version: bool = True) -> Optional[AgentResponse]: ...
    def get_agent(self, agent_id: str) -> Optional[AgentResponse]: ...
    def get_all_agents(self) -> List[AgentResponse]: ...
    def delete_agent(self, agent_id: str) -> bool: ...
    async def regenerate_prompt(self, agent_id: str) -> Optional[AgentResponse]: ...
    def get_runnable_agent(self, agent_id: Optional[str] = None) -> Agent: ...
    async def generate_prompt_background(self, agent_id: str) -> None: ...

# Removed helper functions moved to models/agent.py

# SYSTEM_AGENTS moved to app/services/agents/system_agents.py

class AgentService(IAgentService):
    def __init__(self, llm_service: ILlmService, prompt_writer_service: IPromptWriterAgentService, collection: TenantCollection):
        self.llm_service = llm_service
        self.prompt_writer = prompt_writer_service
        self.collection = collection

    async def create_agent(self, req: AgentCreateRequest) -> AgentResponse:
        new_agent = AgentDocument(
            name=req.name,
            role=req.role,
            goal=req.goal,
            backstory=req.backstory,
            llm_id=req.llm_id,
            system_prompt=None,
            status="pending",
            version=1,
            base_id=None
        )
        
        doc = new_agent.model_dump()
        res = self.collection.insert_one(doc)
        
        base_id_str = str(res.inserted_id)
        self.collection.update_one(
            {"_id": res.inserted_id},
            {"$set": {"base_id": base_id_str}}
        )
        doc["_id"] = res.inserted_id
        doc["base_id"] = base_id_str
        return format_agent_response(doc)

    def _get_agent_doc(self, agent_id: str) -> Optional[dict]:
        base_id, version = parse_agent_id(agent_id)
        if version is not None:
            return self.collection.find_one({"base_id": base_id, "version": version})
        else:
            return self.collection.find_one({"base_id": base_id}, sort=[("version", -1)])

    def get_agent(self, agent_id: str) -> Optional[AgentResponse]:
        if agent_id in SYSTEM_AGENTS:
            return SYSTEM_AGENTS[agent_id]
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return None
        return format_agent_response(doc)

    async def update_agent(self, agent_id: str, req: AgentUpdateRequest, new_version: bool = True) -> Optional[AgentResponse]:
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return None
            
        base_id = doc["base_id"]
        latest_doc = self.collection.find_one({"base_id": base_id}, sort=[("version", -1)])
        
        if new_version:
            new_version_num = latest_doc.get("version", 1) + 1
            updated_data = {
                "name": req.name if req.name is not None else latest_doc["name"],
                "role": req.role if req.role is not None else latest_doc["role"],
                "goal": req.goal if req.goal is not None else latest_doc["goal"],
                "backstory": req.backstory if req.backstory is not None else latest_doc["backstory"],
                "llm_id": req.llm_id if req.llm_id is not None else latest_doc["llm_id"],
                "system_prompt": req.system_prompt if req.system_prompt is not None else latest_doc.get("system_prompt"),
                "status": req.status if req.status is not None else latest_doc.get("status", "completed"),
                "base_id": base_id,
                "version": new_version_num,
                "created_at": latest_doc.get("created_at", datetime.utcnow()),
                "updated_at": datetime.utcnow()
            }
            
            res = self.collection.insert_one(updated_data)
            updated_data["_id"] = res.inserted_id
            return format_agent_response(updated_data)
        else:
            # Update the latest doc in place
            update_fields = {}
            if req.name is not None: update_fields["name"] = req.name
            if req.role is not None: update_fields["role"] = req.role
            if req.goal is not None: update_fields["goal"] = req.goal
            if req.backstory is not None: update_fields["backstory"] = req.backstory
            if req.llm_id is not None: update_fields["llm_id"] = req.llm_id
            if req.system_prompt is not None: update_fields["system_prompt"] = req.system_prompt
            if req.status is not None: update_fields["status"] = req.status
            update_fields["updated_at"] = datetime.utcnow()
            
            self.collection.update_one(
                {"_id": latest_doc["_id"]},
                {"$set": update_fields}
            )
            
            # Fetch updated doc to return correctly
            updated_doc = self.collection.find_one({"_id": latest_doc["_id"]})
            return format_agent_response(updated_doc)

    def get_all_agents(self) -> List[AgentResponse]:
        pipeline = [
            {"$sort": {"version": -1}},
            {"$group": {
                "_id": "$base_id",
                "doc": {"$first": "$$ROOT"}
            }}
        ]
        latest_agents = list(self.collection.aggregate(pipeline))
        db_agents = [format_agent_response(a["doc"]) for a in latest_agents]
        return list(SYSTEM_AGENTS.values()) + db_agents

    def delete_agent(self, agent_id: str) -> bool:
        base_id, _ = parse_agent_id(agent_id)
        res = self.collection.delete_many({"base_id": base_id})
        return res.deleted_count > 0

    async def regenerate_prompt(self, agent_id: str) -> Optional[AgentResponse]:
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return None
            
        req = AgentUpdateRequest(status="pending")
        return await self.update_agent(agent_id, req)

    async def generate_prompt_background(self, agent_id: str) -> None:
        try:
            doc = self._get_agent_doc(agent_id)
            if not doc:
                return

            new_prompt = await self.prompt_writer.generate_prompt(
                role=doc["role"],
                goal=doc["goal"],
                backstory=doc["backstory"],
                llm_id=doc["llm_id"]
            )
            
            self.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "system_prompt": new_prompt,
                    "status": "completed",
                    "updated_at": datetime.utcnow()
                }}
            )
        except Exception as e:
            print(f"Error generating prompt in background: {e}")
            if 'doc' in locals() and doc:
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "status": "error",
                        "updated_at": datetime.utcnow()
                    }}
                )

    def get_runnable_agent(self, agent_id: Optional[str] = None) -> Agent:
        if not agent_id:
            model_name = os.getenv("LLM_MODEL", "x-ai/grok-4.1-fast") # OpenRouter model name
            api_key = os.getenv("OPENROUTER_API_KEY", "")

            model = OpenRouterModel(
                model_name,
                provider=OpenRouterProvider(api_key=api_key),
            )
            
            instructions = get_full_instructions()
            return Agent(
                model=model,
                instructions=instructions,
                toolsets=[memory_toolset]
            )
            
        if agent_id == "agent-creator":
            from app.services.agents.creator_agent import CreatorAgent
            return CreatorAgent()
            
        doc = self._get_agent_doc(agent_id)
        if not doc:
            raise ValueError("Agent not found")
            
        llm = self.llm_service.get_llm(doc["llm_id"])
        if not llm:
            raise ValueError("LLM associated with Agent not found")
            
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model = OpenRouterModel(
            llm.model,
            provider=OpenRouterProvider(api_key=api_key),
        )
        
        db_instructions = doc.get("system_prompt")
        full_instructions = get_full_instructions(db_instructions)
        
        return Agent(
            model=model,
            instructions=full_instructions,
            toolsets=[memory_toolset]
        )
