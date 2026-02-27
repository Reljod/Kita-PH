from bson import ObjectId
from typing import List, Optional, Protocol
from app.models.llm import LlmCreateRequest, LlmResponse, LlmDocument
from app.db import TenantCollection

class ILlmService(Protocol):
    def add_llm(self, req: LlmCreateRequest) -> LlmResponse:
        ...
    def list_llms(self) -> List[LlmResponse]:
        ...
    def get_llm(self, llm_id: str) -> Optional[LlmResponse]:
        ...
    def delete_llm(self, llm_id: str) -> bool:
        ...

def format_llm_response(doc: dict) -> LlmResponse:
    return LlmResponse(
        id=str(doc["_id"]),
        name=doc["name"],
        model=doc["model"],
        provider=doc["provider"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"]
    )

class LlmService(ILlmService):
    def __init__(self, collection: TenantCollection):
        self.collection = collection

    def add_llm(self, req: LlmCreateRequest) -> LlmResponse:
        new_llm = LlmDocument(
            name=req.name,
            model=req.model,
            provider=req.provider
        )
        doc = new_llm.model_dump()
        res = self.collection.insert_one(doc)
        doc["_id"] = res.inserted_id
        return format_llm_response(doc)

    def list_llms(self) -> List[LlmResponse]:
        llms = self.collection.find().sort("created_at", -1)
        return [format_llm_response(l) for l in llms]

    def get_llm(self, llm_id: str) -> Optional[LlmResponse]:
        try:
            obj_id = ObjectId(llm_id)
        except Exception:
            raise ValueError("Invalid LLM ID")
            
        doc = self.collection.find_one({"_id": obj_id})
        if not doc:
            return None
        return format_llm_response(doc)

    def delete_llm(self, llm_id: str) -> bool:
        try:
            obj_id = ObjectId(llm_id)
        except Exception:
            raise ValueError("Invalid LLM ID")
            
        res = self.collection.delete_one({"_id": obj_id})
        return res.deleted_count > 0
