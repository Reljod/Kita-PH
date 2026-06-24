import asyncio
import json
import logging
from bson import ObjectId
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Protocol, AsyncIterator
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest, ChatDocument
from pydantic_core import to_jsonable_python
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import SystemPromptMessage
from app.services.agent_service import IAgentService
from app.services.chat_context_service import ChatContextService
from app.services.chat_archive_service import ChatArchiveService
from app.db import TenantCollection
from app.exceptions import (
    ChatNotFoundError,
    KitaValidationError,
    AgentRunStreamFailedError
)

logger = logging.getLogger(__name__)

class IChatService(Protocol):
    async def create_chat(self, req: ChatCreateRequest, agent_id: Optional[str] = None, status_key: Optional[str] = None) -> ChatResponse:
        ...
    async def continue_chat(self, chat_id: str, req: ChatContinueRequest, agent_id: Optional[str] = None, status_key: Optional[str] = None) -> Optional[ChatResponse]:
        ...
    async def create_chat_stream(self, req: ChatCreateRequest, agent_id: Optional[str] = None, status_key: Optional[str] = None) -> AsyncIterator[dict]:
        ...
    async def continue_chat_stream(self, chat_id: str, req: ChatContinueRequest, agent_id: Optional[str] = None, status_key: Optional[str] = None) -> AsyncIterator[dict]:
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
        chat_context_service: Optional[ChatContextService] = None,
        chat_archive_service: Optional[ChatArchiveService] = None,
    ):
        self.agent_service = agent_service
        self.collection = collection
        self.chat_context_service = chat_context_service
        self.chat_archive_service = chat_archive_service

    def _count_turns(self, messages: list) -> int:
        turn_count = 0
        for msg in messages:
            role = ""
            if isinstance(msg, dict):
                role = msg.get("role", "")
            else:
                role = getattr(msg, "role", "")
            if role == "user":
                turn_count += 1
        return turn_count

    async def _inject_context(self, chat_doc: dict, message_history: list) -> list:
        chat_id = str(chat_doc["_id"])
        extra_messages = []

        if self.chat_archive_service and chat_doc.get("summary"):
            extra_messages.append(
                SystemPromptMessage(content=f"Archived conversation summary:\n{chat_doc['summary']}")
            )

        if self.chat_context_service:
            facts_text = await self.chat_context_service.format_facts_for_prompt(chat_id)
            if facts_text:
                extra_messages.append(SystemPromptMessage(content=facts_text))

        return extra_messages + list(message_history) if extra_messages else list(message_history)

    async def _post_process_run(self, chat_id: str, chat_doc: dict, messages_dump: list):
        turn_count = self._count_turns(messages_dump)
        update_fields = {
            "message_count": turn_count,
            "updated_at": datetime.now(timezone.utc)
        }

        if self.chat_archive_service:
            kept_messages, summary = await self.chat_archive_service.archive_old_messages(
                chat_id, messages_dump, org_id=self.collection.org_id
            )
            if summary:
                update_fields["messages"] = kept_messages
                update_fields["summary"] = summary

        self.collection.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": update_fields}
        )

        asyncio.create_task(self._extract_facts(chat_id, messages_dump))

    async def _extract_facts(self, chat_id: str, messages_dump: list):
        try:
            context_agent = self.agent_service.get_agent_by_name("__system__context_agent")
            if not context_agent:
                return
            serialized = json.dumps(messages_dump, default=str)
            truncated = serialized[:5000]
            prompt = f"Extract important facts from this conversation turn:\n\n{truncated}"
            await self.agent_service.run(
                agent_id=context_agent.id,
                query=prompt,
                chat_id=chat_id,
            )
        except Exception as e:
            logger.warning(f"Fact extraction failed for chat {chat_id}: {e}")

    async def create_chat(self, req: ChatCreateRequest, agent_id: Optional[str] = None, status_key: Optional[str] = None) -> ChatResponse:
        if not agent_id:
            raise KitaValidationError("agent_id is required")

        chat_id = str(ObjectId())
        
        result = await self.agent_service.run(
            agent_id=agent_id,
            query=req.message,
            message_history=None,
            chat_id=chat_id,
            status_key=status_key
        )
        
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
        doc["_id"] = ObjectId(chat_id)
        self.collection.insert_one(doc)
        
        await self._post_process_run(chat_id, doc, messages_dump)
        
        return format_chat_response(doc)

    async def continue_chat(self, chat_id: str, req: ChatContinueRequest, agent_id: Optional[str] = None, status_key: Optional[str] = None) -> Optional[ChatResponse]:
        try:
            obj_id = ObjectId(chat_id)
        except Exception:
            raise ChatNotFoundError(chat_id, message=f"Invalid chat ID: {chat_id}")
            
        chat_query = {"_id": obj_id}
        if agent_id:
            chat_query["agent_id"] = agent_id
            
        chat = self.collection.find_one(chat_query)
        if not chat:
            raise ChatNotFoundError(chat_id)
            
        agent_to_run_with = agent_id or chat.get("agent_id")
        if not agent_to_run_with:
            raise KitaValidationError("agent_id is required")

        # Load history
        message_history = ModelMessagesTypeAdapter.validate_python(chat["messages"])

        # Inject archived summary and KV facts into context
        message_history = await self._inject_context(chat, message_history)

        result = await self.agent_service.run(
            agent_id=agent_to_run_with,
            query=req.message,
            message_history=message_history,
            chat_id=chat_id,
            status_key=status_key
        )
        
        # Dump new history
        messages_dump = to_jsonable_python(result.all_messages())
        
        update_fields = {
            "messages": messages_dump,
            "updated_at": datetime.now(timezone.utc)
        }
        
        if agent_id:
            from app.models.agent import parse_agent_id
            base_agent_id, _ = parse_agent_id(agent_id)
            update_fields["agent_id"] = base_agent_id
            
        self.collection.update_one(
            {"_id": obj_id},
            {"$set": update_fields}
        )

        await self._post_process_run(chat_id, chat, messages_dump)
        
        updated_doc = self.collection.find_one({"_id": obj_id})
        return format_chat_response(updated_doc)

    async def create_chat_stream(
        self,
        req: ChatCreateRequest,
        agent_id: Optional[str] = None,
        status_key: Optional[str] = None
    ) -> AsyncIterator[dict]:
        if not agent_id:
            raise KitaValidationError("agent_id is required")

        chat_id = str(ObjectId())
        
        final_result = None
        async for chunk in self.agent_service.run_stream(
            agent_id=agent_id,
            query=req.message,
            message_history=None,
            chat_id=chat_id,
            status_key=status_key
        ):
            if chunk["type"] == "result":
                final_result = chunk["result"]
            else:
                yield chunk

        if final_result is None:
            raise AgentRunStreamFailedError(agent_id or "unknown", "No run result returned from agent run stream")

        messages_dump = to_jsonable_python(final_result.all_messages())
        
        from app.models.agent import parse_agent_id
        base_agent_id = None
        if agent_id:
            base_agent_id, _ = parse_agent_id(agent_id)
            
        new_chat = ChatDocument(
            messages=messages_dump,
            agent_id=base_agent_id
        )
        
        doc = new_chat.model_dump()
        doc["_id"] = ObjectId(chat_id)
        self.collection.insert_one(doc)
        
        await self._post_process_run(chat_id, doc, messages_dump)
        
        chat_resp = format_chat_response(doc)
        yield {"type": "done", "chat": to_jsonable_python(chat_resp)}

    async def continue_chat_stream(
        self,
        chat_id: str,
        req: ChatContinueRequest,
        agent_id: Optional[str] = None,
        status_key: Optional[str] = None
    ) -> AsyncIterator[dict]:
        try:
            obj_id = ObjectId(chat_id)
        except Exception:
            raise ChatNotFoundError(chat_id, message=f"Invalid chat ID: {chat_id}")
            
        chat_query = {"_id": obj_id}
        if agent_id:
            chat_query["agent_id"] = agent_id
            
        chat = self.collection.find_one(chat_query)
        if not chat:
            raise ChatNotFoundError(chat_id)
            
        agent_to_run_with = agent_id or chat.get("agent_id")
        if not agent_to_run_with:
            raise KitaValidationError("agent_id is required")

        # Load history
        message_history = ModelMessagesTypeAdapter.validate_python(chat["messages"])

        # Inject archived summary and KV facts into context
        message_history = await self._inject_context(chat, message_history)

        final_result = None
        async for chunk in self.agent_service.run_stream(
            agent_id=agent_to_run_with,
            query=req.message,
            message_history=message_history,
            chat_id=chat_id,
            status_key=status_key
        ):
            if chunk["type"] == "result":
                final_result = chunk["result"]
            else:
                yield chunk

        if final_result is None:
            raise AgentRunStreamFailedError(agent_to_run_with or "unknown", "No run result returned from agent run stream")

        # Dump new history
        messages_dump = to_jsonable_python(final_result.all_messages())
        
        update_fields = {
            "messages": messages_dump,
            "updated_at": datetime.now(timezone.utc)
        }
        
        if agent_id:
            from app.models.agent import parse_agent_id
            base_agent_id, _ = parse_agent_id(agent_id)
            update_fields["agent_id"] = base_agent_id
            
        self.collection.update_one(
            {"_id": obj_id},
            {"$set": update_fields}
        )

        await self._post_process_run(chat_id, chat, messages_dump)
        
        updated_doc = self.collection.find_one({"_id": obj_id})
        chat_resp = format_chat_response(updated_doc)
        yield {"type": "done", "chat": to_jsonable_python(chat_resp)}


    def get_chat(self, chat_id: str, agent_id: Optional[str] = None) -> Optional[ChatResponse]:
        try:
            obj_id = ObjectId(chat_id)
        except Exception:
            raise ChatNotFoundError(chat_id, message=f"Invalid chat ID: {chat_id}")
            
        chat_query = {"_id": obj_id}
        if agent_id:
            chat_query["agent_id"] = agent_id
            
        chat = self.collection.find_one(chat_query)
        if not chat:
            raise ChatNotFoundError(chat_id)
            
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
