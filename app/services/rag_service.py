import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from app.models.rag import RagCreateRequest, RagUpdateRequest, RagResponse, RagDocument
from app.db import TenantCollection

class IRagService(Protocol):
    async def add_rag(self, req: RagCreateRequest) -> RagResponse:
        ...
    async def edit_rag(self, rag_id: str, req: RagUpdateRequest) -> Optional[RagResponse]:
        ...
    async def delete_rag(self, rag_id: str) -> bool:
        ...
    def get_rag(self, rag_id: str) -> Optional[RagResponse]:
        ...
    def get_all_rags(self) -> List[RagResponse]:
        ...
    async def update_embedding(self, rag_id: str):
        ...
    async def search(self, query: str, limit: int = 5) -> List[RagResponse]:
        ...

def format_rag_response(doc: Dict[str, Any]) -> RagResponse:
    return RagResponse(
        id=str(doc["_id"]),
        title=doc["title"],
        content=doc["content"],
        status=doc.get("status", "pending"),
        agent_id=doc.get("agent_id"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"]
    )

class MongoVectorDbRagService(IRagService):
    _model = None

    def __init__(self, collection: TenantCollection, agent_id: Optional[str] = None):
        self.collection = collection
        self.agent_id = agent_id

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # We initialize the model lazily
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    async def add_rag(self, req: RagCreateRequest) -> RagResponse:
        new_rag = RagDocument(
            title=req.title,
            content=req.content,
            agent_id=req.agent_id or self.agent_id,
            status="pending"
        )
        doc = new_rag.model_dump()
        res = self.collection.insert_one(doc)
        doc["_id"] = res.inserted_id
        return format_rag_response(doc)

    async def edit_rag(self, rag_id: str, req: RagUpdateRequest) -> Optional[RagResponse]:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")

        update_data = {k: v for k, v in req.model_dump().items() if v is not None}
        if not update_data:
            return self.get_rag(rag_id)

        update_data["updated_at"] = datetime.utcnow()
        # If content changed, we should ideally mark as pending to re-embed
        # For simplicity in this requirement, we'll mark as pending if content is updated
        if "content" in update_data:
            update_data["status"] = "pending"

        query = {"_id": obj_id}
        if self.agent_id:
            query["agent_id"] = self.agent_id

        self.collection.update_one(
            query,
            {"$set": update_data}
        )

        # Trigger re-embedding if content changed (this service doesn't handle background tasks directly, 
        # that's usually the route's job, but it provides the method)
        
        return self.get_rag(rag_id)
    async def delete_rag(self, rag_id: str) -> bool:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")
        
        query = {"_id": obj_id}
        if self.agent_id:
            query["agent_id"] = self.agent_id
            
        res = self.collection.delete_one(query)
        return res.deleted_count > 0

    def get_rag(self, rag_id: str) -> Optional[RagResponse]:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")
            
        query = {"_id": obj_id}
        if self.agent_id:
            query["$or"] = [{"agent_id": self.agent_id}, {"agent_id": None}]
            
        doc = self.collection.find_one(query)
        if not doc:
            return None
            
        return format_rag_response(doc)

    def get_all_rags(self) -> List[RagResponse]:
        query = {}
        if self.agent_id:
            query["$or"] = [{"agent_id": self.agent_id}, {"agent_id": None}]
            
        docs = self.collection.find(query).sort("updated_at", -1)
        return [format_rag_response(d) for d in docs]

    async def update_embedding(self, rag_id: str):
        try:
            obj_id = ObjectId(rag_id)
            doc = self.collection.find_one({"_id": obj_id})
            if not doc:
                return

            model = self.get_model()
            # Run embedding in a thread pool since it's CPU bound
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, model.encode, doc["content"])
            
            self.collection.update_one(
                {"_id": obj_id},
                {
                    "$set": {
                        "embedding": embedding.tolist(),
                        "status": "completed",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        except Exception as e:
            print(f"Error updating embedding for {rag_id}: {e}")
            try:
                self.collection.update_one(
                    {"_id": ObjectId(rag_id)},
                    {"$set": {"status": "error", "updated_at": datetime.utcnow()}}
                )
            except:
                pass

    async def search(self, query: str, limit: int = 5) -> List[RagResponse]:
        model = self.get_model()
        # Run embedding in a thread pool since it's CPU bound
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(None, model.encode, query)
        
        # Atlas Vector Search
        # Note: This requires a vector index named 'vector_index' on the collection
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "rag_index",
                    "path": "embedding",
                    "queryVector": query_embedding.tolist(),
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            }
        ]
        
        try:
            # If we have an agent_id, we need to filter results by it or org-wide
            if self.agent_id:
                pipeline.append({
                    "$match": {
                        "$or": [{"agent_id": self.agent_id}, {"agent_id": None}]
                    }
                })
            
            docs = list(self.collection.aggregate(pipeline))
            return [format_rag_response(d) for d in docs]
        except Exception as e:
            print(f"Error during vector search: {e}")
            # Fallback to simple text search or returning empty list if vector search index is not set up
            return []
