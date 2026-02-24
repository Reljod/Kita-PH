from bson import ObjectId
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from app.models.chat import ChatCreateRequest, ChatResponse, ChatContinueRequest, ChatDocument
from pydantic_core import to_jsonable_python
from pydantic_ai import ModelMessagesTypeAdapter
from app.db import db
from app.services.agent_service import IAgentService

class IChatService(Protocol):
    async def create_chat(self, req: ChatCreateRequest) -> ChatResponse:
        ...
    async def continue_chat(self, chat_id: str, req: ChatContinueRequest) -> Optional[ChatResponse]:
        ...
    def get_chat(self, chat_id: str) -> Optional[ChatResponse]:
        ...
    def get_all_chats(self) -> List[ChatResponse]:
        ...

def format_chat_response(doc: Dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        id=str(doc["_id"]),
        messages=doc["messages"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"]
    )

class ChatService(IChatService):
    def __init__(self, agent_service: IAgentService):
        self.agent_service = agent_service

    async def create_chat(self, req: ChatCreateRequest) -> ChatResponse:
        agent = self.agent_service.get_runnable_agent()
        result = await agent.run(req.message)
        
        messages_dump = to_jsonable_python(result.all_messages())
        
        new_chat = ChatDocument(
            messages=messages_dump
        )
        
        doc = new_chat.model_dump()
        res = db.get_chats_collection().insert_one(doc)
        doc["_id"] = res.inserted_id
        
        return format_chat_response(doc)

    async def continue_chat(self, chat_id: str, req: ChatContinueRequest) -> Optional[ChatResponse]:
        try:
            obj_id = ObjectId(chat_id)
        except Exception:
            raise ValueError("Invalid chat ID")
            
        chat = db.get_chats_collection().find_one({"_id": obj_id})
        if not chat:
            return None
            
        agent = self.agent_service.get_runnable_agent()
        
        # Load history
        message_history = ModelMessagesTypeAdapter.validate_python(chat["messages"])
        
        result = await agent.run(req.message, message_history=message_history)
        
        # Dump new history
        messages_dump = to_jsonable_python(result.all_messages())
        
        db.get_chats_collection().update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "messages": messages_dump,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        updated_chat = db.get_chats_collection().find_one({"_id": obj_id})
        return format_chat_response(updated_chat)

    def get_chat(self, chat_id: str) -> Optional[ChatResponse]:
        try:
            obj_id = ObjectId(chat_id)
        except Exception:
            raise ValueError("Invalid chat ID")
            
        chat = db.get_chats_collection().find_one({"_id": obj_id})
        if not chat:
            return None
            
        return format_chat_response(chat)

    def get_all_chats(self) -> List[ChatResponse]:
        chats = db.get_chats_collection().find().sort("updated_at", -1)
        return [format_chat_response(c) for c in chats]
