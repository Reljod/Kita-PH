from bson import ObjectId
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest, ChatDocument
from pydantic_core import to_jsonable_python
from pydantic_ai import ModelMessagesTypeAdapter
from app.services.agent_service import IAgentService
from app.db import TenantCollection

class IChatService(Protocol):
    async def create_chat(self, req: ChatCreateRequest, agent_id: Optional[str] = None) -> ChatResponse:
        ...
    async def continue_chat(self, chat_id: str, req: ChatContinueRequest, agent_id: Optional[str] = None) -> Optional[ChatResponse]:
        ...
    def get_chat(self, chat_id: str, agent_id: Optional[str] = None) -> Optional[ChatResponse]:
        ...
    def get_all_chats(self, agent_id: Optional[str] = None, preview: bool = False, limit: Optional[int] = None) -> List[ChatResponse]:
        ...

def format_chat_response(doc: Dict[str, Any], preview_only: bool = False) -> ChatResponse:
    messages = doc["messages"]
    preview_text = None
    
    if messages:
        # Get the first user message content for preview
        first_msg = messages[0]
        content = ""
        if isinstance(first_msg, dict):
            if "parts" in first_msg:
                content = "\n".join([p["content"] for p in first_msg["parts"] if p.get("part_kind") != "thinking"])
            else:
                content = first_msg.get("content", "")
        
        preview_text = content[:100] + "..." if len(content) > 100 else content
        
        if preview_only:
            messages = [first_msg] # Only return the first message if previewing

    return ChatResponse(
        id=str(doc["_id"]),
        messages=messages,
        agent_id=doc.get("agent_id"),
        preview=preview_text,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"]
    )

class ChatService(IChatService):
    def __init__(
        self, 
        agent_service: IAgentService, 
        collection: TenantCollection,
        file_service: Optional[Any] = None,
        parse_service: Optional[Any] = None,
        graph_rag_service: Optional[Any] = None,
        rag_service: Optional[Any] = None
    ):
        self.agent_service = agent_service
        self.collection = collection
        self.file_service = file_service
        self.parse_service = parse_service
        self.graph_rag_service = graph_rag_service
        self.rag_service = rag_service

    def _get_deps(self, agent_id: Optional[str]) -> Dict[str, Any]:
        from app.services.rag_service import MongoVectorDbRagService
        from app.db import db
        
        # Build scoped rag_service
        rag_service = self.rag_service
        if not rag_service:
            # Fallback construct
            rag_coll = TenantCollection(db.get_rag_collection(), self.collection.org_id)
            rag_service = MongoVectorDbRagService(rag_coll, agent_id=agent_id)
        else:
            # Create a copy/instance bound to agent_id
            rag_service = MongoVectorDbRagService(rag_service.collection, agent_id=agent_id)

        return {
            "org_id": self.collection.org_id, 
            "agent_id": agent_id, 
            "agent_service": self.agent_service,
            "file_service": self.file_service,
            "parse_service": self.parse_service,
            "graph_rag_service": self.graph_rag_service,
            "rag_service": rag_service
        }

    async def create_chat(self, req: ChatCreateRequest, agent_id: Optional[str] = None) -> ChatResponse:
        if not agent_id:
            raise ValueError("agent_id is required")

        agent = self.agent_service.get_runnable_agent(agent_id=agent_id)
        result = await agent.run(req.message, deps=self._get_deps(agent_id))
        
        messages_dump = to_jsonable_python(result.all_messages())
        
        # Determine the agent's actual ID (system agents might have static IDs)
        
        from app.models.agent import parse_agent_id
        base_agent_id = None
        if agent_id:
            base_agent_id, _ = parse_agent_id(agent_id)
            
        new_chat = ChatDocument(
            messages=messages_dump,
            agent_id=base_agent_id
        )
        
        doc = new_chat.model_dump()
        res = self.collection.insert_one(doc)
        doc["_id"] = res.inserted_id
        
        return format_chat_response(doc)

    async def continue_chat(self, chat_id: str, req: ChatContinueRequest, agent_id: Optional[str] = None) -> Optional[ChatResponse]:
        try:
            obj_id = ObjectId(chat_id)
        except Exception:
            raise ValueError("Invalid chat ID")
            
        chat_query = {"_id": obj_id}
        if agent_id:
            chat_query["agent_id"] = agent_id
            
        chat = self.collection.find_one(chat_query)
        if not chat:
            return None
            
        agent_to_run_with = agent_id or chat.get("agent_id")
        if not agent_to_run_with:
            raise ValueError("agent_id is required")

        agent = self.agent_service.get_runnable_agent(agent_id=agent_to_run_with)
        
        # Load history
        message_history = ModelMessagesTypeAdapter.validate_python(chat["messages"])
        
        result = await agent.run(req.message, message_history=message_history, deps=self._get_deps(agent_to_run_with))
        
        # Dump new history
        messages_dump = to_jsonable_python(result.all_messages())
        
        update_fields = {
            "messages": messages_dump,
            "updated_at": datetime.utcnow()
        }
        
        if agent_id:
            from app.models.agent import parse_agent_id
            base_agent_id, _ = parse_agent_id(agent_id)
            update_fields["agent_id"] = base_agent_id
            
        self.collection.update_one(
            {"_id": obj_id},
            {"$set": update_fields}
        )
        
        updated_doc = self.collection.find_one({"_id": obj_id})
        return format_chat_response(updated_doc)

    def get_chat(self, chat_id: str, agent_id: Optional[str] = None) -> Optional[ChatResponse]:
        try:
            obj_id = ObjectId(chat_id)
        except Exception:
            raise ValueError("Invalid chat ID")
            
        chat_query = {"_id": obj_id}
        if agent_id:
            chat_query["agent_id"] = agent_id
            
        chat = self.collection.find_one(chat_query)
        if not chat:
            return None
            
        return format_chat_response(chat)

    def get_all_chats(self, agent_id: Optional[str] = None, preview: bool = False, limit: Optional[int] = None) -> List[ChatResponse]:
        query = {}
        if agent_id:
            from app.models.agent import parse_agent_id
            base_id, _ = parse_agent_id(agent_id)
            query["agent_id"] = base_id
        
        cursor = self.collection.find(query).sort("updated_at", -1)
        if limit:
            cursor = cursor.limit(limit)
            
        return [format_chat_response(c, preview_only=preview) for c in cursor]
