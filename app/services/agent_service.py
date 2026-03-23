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
from app.services.agents.system_agents import SYSTEM_AGENTS, BASE_AGENT
from app.services.agents.templates.system_prompt import build_system_prompt
from app.services.tools.memory_tools import memory_toolset
from app.services.tools import get_toolsets_by_names
from app.db import TenantCollection


class IAgentService(Protocol):
    async def create_agent(self, req: AgentCreateRequest) -> AgentResponse: ...
    async def update_agent(self, agent_id: str, req: AgentUpdateRequest, new_version: bool = True) -> Optional[AgentResponse]: ...
    def get_agent(self, agent_id: str) -> Optional[AgentResponse]: ...
    def get_all_agents(self, include_last_chat: bool = False) -> List[AgentResponse]: ...
    def delete_agent(self, agent_id: str) -> bool: ...
    def get_runnable_agent(self, agent_id: Optional[str] = None) -> Agent: ...
    async def add_tools(self, agent_id: str, tool_ids: List[str]) -> bool: ...
    async def remove_tools(self, agent_id: str, tool_ids: List[str]) -> bool: ...
    def get_agents_by_tool(self, tool_id: str) -> List[AgentResponse]: ...


class AgentService(IAgentService):
    def __init__(self, llm_service: ILlmService, collection: TenantCollection, tools_collection: Optional[TenantCollection] = None):
        self.llm_service = llm_service
        self.collection = collection
        self.tools_collection = tools_collection

    async def create_agent(self, req: AgentCreateRequest) -> AgentResponse:
        new_agent = AgentDocument(
            name=req.name,
            role=req.role,
            goal=req.goal,
            backstory=req.backstory,
            personalities=req.personalities,
            llm_id=req.llm_id,
            tools=req.tools or [],
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

        system_prompt = build_system_prompt(
            name=req.name,
            role=req.role,
            goal=req.goal,
            backstory=req.backstory,
            personalities=req.personalities,
        )
        return format_agent_response(doc, system_prompt=system_prompt)

    def _get_agent_doc(self, agent_id: str) -> Optional[dict]:
        base_id, version = parse_agent_id(agent_id)
        if version is not None:
            return self.collection.find_one({"base_id": base_id, "version": version})
        else:
            return self.collection.find_one({"base_id": base_id}, sort=[("version", -1)])

    def _build_prompt_from_doc(self, doc: dict) -> str:
        return build_system_prompt(
            name=doc["name"],
            role=doc["role"],
            goal=doc["goal"],
            backstory=doc["backstory"],
            personalities=doc.get("personalities"),
        )

    def get_agent(self, agent_id: str) -> Optional[AgentResponse]:
        if agent_id in SYSTEM_AGENTS:
            return SYSTEM_AGENTS[agent_id]
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return None
        system_prompt = self._build_prompt_from_doc(doc)
        return format_agent_response(doc, system_prompt=system_prompt)

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
                "personalities": req.personalities if req.personalities is not None else latest_doc.get("personalities"),
                "llm_id": req.llm_id if req.llm_id is not None else latest_doc["llm_id"],
                "tools": req.tools if req.tools is not None else latest_doc.get("tools", []),
                "base_id": base_id,
                "version": new_version_num,
                "created_at": latest_doc.get("created_at", datetime.utcnow()),
                "updated_at": datetime.utcnow()
            }
            
            res = self.collection.insert_one(updated_data)
            updated_data["_id"] = res.inserted_id
            system_prompt = self._build_prompt_from_doc(updated_data)
            return format_agent_response(updated_data, system_prompt=system_prompt)
        else:
            update_fields = {}
            if req.name is not None: update_fields["name"] = req.name
            if req.role is not None: update_fields["role"] = req.role
            if req.goal is not None: update_fields["goal"] = req.goal
            if req.backstory is not None: update_fields["backstory"] = req.backstory
            if req.personalities is not None: update_fields["personalities"] = req.personalities
            if req.llm_id is not None: update_fields["llm_id"] = req.llm_id
            if req.tools is not None: update_fields["tools"] = req.tools
            update_fields["updated_at"] = datetime.utcnow()
            
            self.collection.update_one(
                {"_id": latest_doc["_id"]},
                {"$set": update_fields}
            )
            
            updated_doc = self.collection.find_one({"_id": latest_doc["_id"]})
            system_prompt = self._build_prompt_from_doc(updated_doc)
            return format_agent_response(updated_doc, system_prompt=system_prompt)

    def get_all_agents(self, include_last_chat: bool = False) -> List[AgentResponse]:
        pipeline = [
            {"$sort": {"version": -1}},
            {"$group": {
                "_id": "$base_id",
                "doc": {"$first": "$$ROOT"}
            }}
        ]
        latest_agents = list(self.collection.aggregate(pipeline))
        # System prompt is omitted in list view for performance
        db_agents = [format_agent_response(a["doc"]) for a in latest_agents]
        all_agents = list(SYSTEM_AGENTS.values()) + db_agents
        
        if include_last_chat:
            from app.services.chat_service import ChatService
            from app.db import db
            chats_coll = TenantCollection(db.get_chats_collection(), self.collection.org_id)
            chat_service = ChatService(self, chats_coll)
            
            for agent in all_agents:
                last_chats = chat_service.get_all_chats(agent_id=agent.id, preview=True, limit=1)
                if last_chats:
                    agent.last_chat = last_chats[0]
                    
        return all_agents

    def delete_agent(self, agent_id: str) -> bool:
        base_id, _ = parse_agent_id(agent_id)
        res = self.collection.delete_many({"base_id": base_id})
        return res.deleted_count > 0

    async def add_tools(self, agent_id: str, tool_ids: List[str]) -> bool:
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return False
        
        self.collection.update_one(
            {"_id": doc["_id"]},
            {
                "$addToSet": {"tools": {"$each": tool_ids}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return True

    async def remove_tools(self, agent_id: str, tool_ids: List[str]) -> bool:
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return False
        
        self.collection.update_one(
            {"_id": doc["_id"]},
            {
                "$pull": {"tools": {"$in": tool_ids}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return True

    def get_agents_by_tool(self, tool_id: str) -> List[AgentResponse]:
        # tool_id could be Mongo ID or tool name
        query = {"tools": tool_id}
        
        pipeline = [
            {"$match": query},
            {"$sort": {"version": -1}},
            {"$group": {
                "_id": "$base_id",
                "doc": {"$first": "$$ROOT"}
            }}
        ]
        latest_agents = list(self.collection.aggregate(pipeline))
        return [format_agent_response(a["doc"]) for a in latest_agents]

    def get_runnable_agent(self, agent_id: Optional[str] = None) -> Agent:
        if not agent_id:
            model_name = os.getenv("LLM_MODEL", "x-ai/grok-4.1-fast")
            api_key = os.getenv("OPENROUTER_API_KEY", "")

            model = OpenRouterModel(
                model_name,
                provider=OpenRouterProvider(api_key=api_key),
            )
            
            instructions = build_system_prompt(
                name=BASE_AGENT.name,
                role=BASE_AGENT.role,
                goal=BASE_AGENT.goal,
                backstory=BASE_AGENT.backstory,
                personalities=BASE_AGENT.personalities
            )
            return Agent(
                model=model,
                instructions=instructions,
                toolsets=[memory_toolset]
            )
            
        if agent_id in SYSTEM_AGENTS:
            agent_def = SYSTEM_AGENTS[agent_id]
            # Creator agent has special logic
            if agent_id == "agent-creator":
                from app.services.agents.creator_agent import CreatorAgent
                return CreatorAgent()
            
            # Others use the standard template
            model_name = agent_def.llm_id or os.getenv("LLM_MODEL", "x-ai/grok-4.1-fast")
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            model = OpenRouterModel(
                model_name,
                provider=OpenRouterProvider(api_key=api_key),
            )
            instructions = build_system_prompt(
                name=agent_def.name,
                role=agent_def.role,
                goal=agent_def.goal,
                backstory=agent_def.backstory,
                personalities=agent_def.personalities
            )
            return Agent(
                model=model,
                instructions=instructions,
                toolsets=[memory_toolset]
            )

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
        
        system_prompt = self._build_prompt_from_doc(doc)
        
        tool_ids = doc.get("tools", [])
        tool_names = []
        
        if self.tools_collection and tool_ids:
            for tid in tool_ids:
                try:
                    t_doc = self.tools_collection.find_one({"_id": ObjectId(tid)})
                    if t_doc:
                        tool_names.append(t_doc["name"])
                except Exception:
                    # Fallback: check if tid is actually the name
                    t_doc = self.tools_collection.find_one({"name": tid})
                    if t_doc:
                        tool_names.append(t_doc["name"])

        dynamic_toolsets = get_toolsets_by_names(tool_names)
        
        return Agent(
            model=model,
            instructions=system_prompt,
            toolsets=dynamic_toolsets
        )
