import os
import logging
from bson import ObjectId
import pymongo
from typing import List, Optional, Protocol, Any, AsyncIterator
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from app.models.agent import (
    AgentCreateRequest, AgentUpdateRequest, AgentResponse, AgentDocument,
    parse_agent_id, format_agent_response
)
from app.services.llm_service import ILlmService
from app.services.agents.templates.system_prompt import build_system_prompt
from app.services.tools.memory_tools import memory_toolset
from app.services.tools.delegation_tools import delegation_toolset
from app.services.tools import get_tools_by_names
from app.db import TenantCollection
logger = logging.getLogger(__name__)


class IAgentService(Protocol):
    async def create_agent(self, req: AgentCreateRequest) -> AgentResponse: ...
    async def update_agent(self, agent_id: str, req: AgentUpdateRequest, new_version: bool = True) -> Optional[AgentResponse]: ...
    def get_agent(self, agent_id: str) -> Optional[AgentResponse]: ...
    def get_all_agents(self, include_last_chat: bool = False) -> List[AgentResponse]: ...
    def delete_agent(self, agent_id: str) -> bool: ...
    def get_runnable_agent(self, agent_id: str) -> Agent: ...
    async def add_tools(self, agent_id: str, tool_ids: List[str]) -> bool: ...
    async def remove_tools(self, agent_id: str, tool_ids: List[str]) -> bool: ...
    def get_agents_by_tool(self, tool_id: str) -> List[AgentResponse]: ...
    async def run(
        self,
        agent_id: str,
        query: str,
        message_history: Optional[List[Any]] = None,
        chat_id: Optional[str] = None,
        status_key: Optional[str] = None
    ) -> Any: ...
    async def run_stream(
        self,
        agent_id: str,
        query: str,
        message_history: Optional[List[Any]] = None,
        chat_id: Optional[str] = None,
        status_key: Optional[str] = None
    ) -> AsyncIterator[dict]: ...



class AgentService(IAgentService):
    def __init__(self, llm_service: ILlmService, collection: TenantCollection, tools_collection: Optional[TenantCollection] = None):
        self.llm_service = llm_service
        self.collection = collection
        self.tools_collection = tools_collection
        # Counters collection shares the same DB as the agents collection
        self._counters = collection._collection.database["version_counters"]

    def _next_version(self, base_id: str) -> int:
        """Atomically allocate the next version number for a given base_id.
        
        Uses find_one_and_update with $inc so concurrent callers can never
        receive the same version number.
        """
        result = self._counters.find_one_and_update(
            {"_id": base_id},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=pymongo.ReturnDocument.AFTER
        )
        return result["seq"]

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

        tool_ids = req.tools or []
        tool_names = self._resolve_tool_names(tool_ids)
        system_prompt = build_system_prompt(
            name=req.name,
            role=req.role,
            goal=req.goal,
            backstory=req.backstory,
            personalities=req.personalities,
            tools=tool_names,
        )
        return format_agent_response(doc, system_prompt=system_prompt)

    def _get_agent_doc(self, agent_id: str) -> Optional[dict]:
        base_id, version = parse_agent_id(agent_id)
        if version is not None:
            return self.collection.find_one({"base_id": base_id, "version": version})
        else:
            return self.collection.find_one({"base_id": base_id}, sort=[("version", -1)])

    def _resolve_tool_names(self, tool_ids: List[str]) -> List[str]:
        if not tool_ids:
            tool_ids = []
        tool_names = []
        if self.tools_collection:
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
        else:
            # Fallback if no tools collection is available
            tool_names = [str(tid) for tid in tool_ids]
            
        if "delegate_task" not in tool_names:
            tool_names.append("delegate_task")
            
        return tool_names

    def _build_prompt_from_doc(self, doc: dict) -> str:
        tool_ids = doc.get("tools", [])
        tool_names = self._resolve_tool_names(tool_ids)
        return build_system_prompt(
            name=doc["name"],
            role=doc["role"],
            goal=doc["goal"],
            backstory=doc["backstory"],
            personalities=doc.get("personalities"),
            tools=tool_names,
        )

    def get_agent(self, agent_id: str) -> Optional[AgentResponse]:
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
            new_version_num = self._next_version(base_id)
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
                "created_at": latest_doc.get("created_at", datetime.now(timezone.utc)),
                "updated_at": datetime.now(timezone.utc)
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
            update_fields["updated_at"] = datetime.now(timezone.utc)
            
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
        
        if include_last_chat:
            from app.services.chat_service import ChatService
            from app.db import db
            chats_coll = TenantCollection(db.get_chats_collection(), self.collection.org_id)
            chat_service = ChatService(self, chats_coll)
            
            for agent in db_agents:
                last_chats = chat_service.get_all_chats(agent_id=agent.id, preview=True, limit=1)
                if last_chats:
                    agent.last_chat = last_chats[0]
                    
        return db_agents

    def delete_agent(self, agent_id: str) -> bool:
        base_id, _ = parse_agent_id(agent_id)
        res = self.collection.delete_many({"base_id": base_id})
        return res.deleted_count > 0

    async def add_tools(self, agent_id: str, tool_ids: List[str]) -> bool:
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return False
        
        base_id = doc["base_id"]
        latest_doc = self.collection.find_one({"base_id": base_id}, sort=[("version", -1)])
        new_version_num = self._next_version(base_id)
        
        current_tools = list(latest_doc.get("tools", []))
        for tid in tool_ids:
            if tid not in current_tools:
                current_tools.append(tid)
                
        updated_data = {
            "name": latest_doc["name"],
            "role": latest_doc["role"],
            "goal": latest_doc["goal"],
            "backstory": latest_doc["backstory"],
            "personalities": latest_doc.get("personalities"),
            "llm_id": latest_doc["llm_id"],
            "tools": current_tools,
            "base_id": base_id,
            "version": new_version_num,
            "created_at": latest_doc.get("created_at", datetime.now(timezone.utc)),
            "updated_at": datetime.now(timezone.utc)
        }
        
        self.collection.insert_one(updated_data)
        return True

    async def remove_tools(self, agent_id: str, tool_ids: List[str]) -> bool:
        doc = self._get_agent_doc(agent_id)
        if not doc:
            return False
        
        base_id = doc["base_id"]
        latest_doc = self.collection.find_one({"base_id": base_id}, sort=[("version", -1)])
        new_version_num = self._next_version(base_id)
        
        current_tools = [t for t in latest_doc.get("tools", []) if t not in tool_ids]
                
        updated_data = {
            "name": latest_doc["name"],
            "role": latest_doc["role"],
            "goal": latest_doc["goal"],
            "backstory": latest_doc["backstory"],
            "personalities": latest_doc.get("personalities"),
            "llm_id": latest_doc["llm_id"],
            "tools": current_tools,
            "base_id": base_id,
            "version": new_version_num,
            "created_at": latest_doc.get("created_at", datetime.now(timezone.utc)),
            "updated_at": datetime.now(timezone.utc)
        }
        
        self.collection.insert_one(updated_data)
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

    def get_runnable_agent(self, agent_id: str) -> Agent:
        doc = self._get_agent_doc(agent_id)
        if not doc:
            raise ValueError(f"Agent '{agent_id}' not found")

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
        tool_names = self._resolve_tool_names(tool_ids)

        # Retrieve only individual requested Tools (excluding delegate_task as it's passed via toolsets)
        dynamic_tools = get_tools_by_names([t for t in tool_names if t != "delegate_task"])

        return Agent(
            model=model,
            instructions=system_prompt,
            tools=dynamic_tools,
            toolsets=[delegation_toolset]
        )

    def _get_deps(self, agent_id: str, status_key: Optional[str] = None) -> dict[str, Any]:
        from app.dependencies.services import get_services
        services = get_services(self.collection.org_id)

        return {
            "org_id": self.collection.org_id, 
            "agent_id": agent_id, 
            "status_key": status_key,
            "agent_service": self,
            "file_service": services.file_service,
            "parse_service": services.parse_service,
            "graph_rag_service": services.graph_rag_service,
            "rag_service": services.rag_service,
            "retrieval_service": services.retrieval_service
        }

    async def run(
        self,
        agent_id: str,
        query: str,
        message_history: Optional[List[Any]] = None,
        chat_id: Optional[str] = None,
        status_key: Optional[str] = None
    ) -> Any:
        from app.dependencies.services import get_services
        services = get_services(self.collection.org_id)

        if status_key:
            await services.agent_status_service.start_session(
                status_key=status_key,
                agent_id=agent_id,
                chat_id=chat_id
            )

        import time
        import logfire
        start_time = time.perf_counter()
        truncated_query = query[:150] + "..." if len(query) > 150 else query

        with logfire.span("agent_run", agent_id=agent_id, query=truncated_query) as span:
            try:
                agent = self.get_runnable_agent(agent_id=agent_id)
                deps = self._get_deps(agent_id, status_key=status_key)

                if status_key:
                    await services.agent_status_service.update_step(status_key, "draft_response", agent_id)

                result = await agent.run(
                    query,
                    message_history=message_history,
                    deps=deps
                )

                duration = time.perf_counter() - start_time
                usage = result.usage()
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("request_tokens", usage.request_tokens)
                span.set_attribute("response_tokens", usage.response_tokens)
                span.set_attribute("total_tokens", usage.total_tokens)
                span.set_attribute("requests", usage.requests)

                logger.info(
                    f"Agent run completed: agent_id={agent_id}, duration={duration:.3f}s, tokens={usage.total_tokens}",
                    extra={
                        "agent_id": agent_id,
                        "duration": duration,
                        "query": truncated_query,
                        "request_tokens": usage.request_tokens,
                        "response_tokens": usage.response_tokens,
                        "total_tokens": usage.total_tokens,
                        "requests": usage.requests,
                    }
                )

                if status_key:
                    await services.agent_status_service.update_step(status_key, "finalize_response", agent_id)

                if status_key:
                    await services.agent_status_service.finish_session(
                        status_key=status_key,
                        success=True,
                        chat_id=chat_id
                    )
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("error", str(e))
                logger.error(
                    f"Agent run failed: agent_id={agent_id}, duration={duration:.3f}s: {e}",
                    extra={
                        "agent_id": agent_id,
                        "duration": duration,
                        "query": truncated_query,
                        "error": str(e),
                    },
                    exc_info=True
                )
                if status_key:
                    await services.agent_status_service.finish_session(
                        status_key=status_key,
                        success=False,
                        chat_id=chat_id
                    )
                raise e

    async def run_stream(
        self,
        agent_id: str,
        query: str,
        message_history: Optional[List[Any]] = None,
        chat_id: Optional[str] = None,
        status_key: Optional[str] = None
    ) -> AsyncIterator[dict]:
        from app.dependencies.services import get_services
        services = get_services(self.collection.org_id)

        if status_key:
            await services.agent_status_service.start_session(
                status_key=status_key,
                agent_id=agent_id,
                chat_id=chat_id
            )

        import time
        import logfire
        start_time = time.perf_counter()
        truncated_query = query[:150] + "..." if len(query) > 150 else query

        with logfire.span("agent_run_stream", agent_id=agent_id, query=truncated_query) as span:
            try:
                agent = self.get_runnable_agent(agent_id=agent_id)
                deps = self._get_deps(agent_id, status_key=status_key)

                if status_key:
                    await services.agent_status_service.update_step(status_key, "draft_response", agent_id)

                has_updated_status = False
                current_run_text = ""
                current_run_has_tools = False

                async for event in agent.run_stream_events(
                    query,
                    message_history=message_history,
                    deps=deps
                ):
                    event_type = type(event).__name__
                    
                    if event_type == "PartStartEvent":
                        if hasattr(event, "part"):
                            part_type = type(event.part).__name__
                            if part_type == "TextPart":
                                chunk = event.part.content
                                current_run_text += chunk
                                if not has_updated_status and status_key:
                                    await services.agent_status_service.update_step(status_key, "finalize_response", agent_id)
                                    has_updated_status = True
                                yield {"type": "content", "delta": chunk}
                            elif part_type == "ThinkingPart":
                                yield {"type": "thought", "delta": event.part.content}
                            elif part_type == "ToolCallPart":
                                current_run_has_tools = True
                                if current_run_text:
                                    yield {"type": "reset"}
                                    yield {"type": "thought", "delta": current_run_text}
                                    current_run_text = ""
                                    
                    elif event_type == "PartDeltaEvent":
                        if hasattr(event, "delta"):
                            delta_type = type(event.delta).__name__
                            if delta_type == "TextPartDelta":
                                chunk = event.delta.content_delta
                                current_run_text += chunk
                                if not has_updated_status and status_key:
                                    await services.agent_status_service.update_step(status_key, "finalize_response", agent_id)
                                    has_updated_status = True
                                yield {"type": "content", "delta": chunk}
                            elif delta_type == "ThinkingPartDelta":
                                yield {"type": "thought", "delta": event.delta.content_delta}
                            elif delta_type == "ToolCallPartDelta":
                                current_run_has_tools = True
                                if current_run_text:
                                    yield {"type": "reset"}
                                    yield {"type": "thought", "delta": current_run_text}
                                    current_run_text = ""
                                    
                    elif event_type == "FunctionToolCallEvent":
                        current_run_has_tools = True
                        if current_run_text:
                            yield {"type": "reset"}
                            yield {"type": "thought", "delta": current_run_text}
                            current_run_text = ""
                            
                    elif event_type == "ModelResponseStreamEvent":
                        current_run_text = ""
                        current_run_has_tools = False
                        
                    elif event_type == "AgentRunResultEvent":
                        duration = time.perf_counter() - start_time
                        result = event.result
                        usage = result.usage()
                        span.set_attribute("duration_seconds", duration)
                        span.set_attribute("request_tokens", usage.request_tokens)
                        span.set_attribute("response_tokens", usage.response_tokens)
                        span.set_attribute("total_tokens", usage.total_tokens)
                        span.set_attribute("requests", usage.requests)

                        logger.info(
                            f"Agent run stream completed: agent_id={agent_id}, duration={duration:.3f}s, tokens={usage.total_tokens}",
                            extra={
                                "agent_id": agent_id,
                                "duration": duration,
                                "query": truncated_query,
                                "request_tokens": usage.request_tokens,
                                "response_tokens": usage.response_tokens,
                                "total_tokens": usage.total_tokens,
                                "requests": usage.requests,
                            }
                        )
                        yield {"type": "result", "result": event.result}

                if status_key:
                    await services.agent_status_service.finish_session(
                        status_key=status_key,
                        success=True,
                        chat_id=chat_id
                    )
            except Exception as e:
                duration = time.perf_counter() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("error", str(e))
                logger.error(
                    f"Agent run stream failed: agent_id={agent_id}, duration={duration:.3f}s: {e}",
                    extra={
                        "agent_id": agent_id,
                        "duration": duration,
                        "query": truncated_query,
                        "error": str(e),
                    },
                    exc_info=True
                )
                if status_key:
                    await services.agent_status_service.finish_session(
                        status_key=status_key,
                        success=False,
                        chat_id=chat_id
                    )
                raise e


