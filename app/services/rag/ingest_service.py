import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Protocol
from bson import ObjectId
from app.models.rag import RagCreateRequest, RagUpdateRequest, RagResponse, RagDocument
from app.db import TenantCollection
from app.services.rag.nested_data_enrichment_service import NestedDataEnrichmentService
from app.services.rag.mongodb_vector_search_rag_service import MongoDBVectorSearchRagService

class IIngestService(Protocol):
    collection: TenantCollection
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
    async def ingest_file_parse(self, file_id: str, org_id: str) -> bool:
        ...

class IngestService(IIngestService):
    def __init__(
        self, 
        collection: TenantCollection, 
        vector_service: MongoDBVectorSearchRagService,
        parse_collection: Optional[TenantCollection] = None,
        agent_id: Optional[str] = None
    ):
        self.collection = collection # file_parsed_flattened
        self.vector_service = vector_service
        self.parse_collection = parse_collection # file_parse
        self.agent_id = agent_id

    def _get_agent_filter(self) -> dict:
        if not self.agent_id:
            return {}
        from app.models.agent import parse_agent_id
        base_id = parse_agent_id(self.agent_id)[0]
        return {
            "$or": [
                {"agent_id": self.agent_id},
                {"agent_id": base_id},
                {"agent_id": {"$regex": f"^{base_id}(-v\\d+)?$"}}
            ]
        }

    async def add_rag(self, req: RagCreateRequest) -> RagResponse:
        from app.models.agent import parse_agent_id
        raw_agent = req.agent_id or self.agent_id
        agent_id = parse_agent_id(raw_agent)[0] if raw_agent else None

        new_doc = {
            "title": req.title,
            "content": req.content,
            "text": f"{req.title}: {req.content}",
            "heading_to_text": f"{req.title} > {req.content}",
            "json_path": "manual_input",
            "breadcrumb": "manual_input",
            "heading_text": req.title,
            "page": 1,
            "location": None,
            "agent_id": agent_id,
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        res = self.collection.insert_one(new_doc)
        new_doc["_id"] = res.inserted_id
        return format_rag_response(new_doc)

    async def edit_rag(self, rag_id: str, req: RagUpdateRequest) -> Optional[RagResponse]:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")

        update_data = {k: v for k, v in req.model_dump().items() if v is not None}
        if not update_data:
            return self.get_rag(rag_id)

        update_data["updated_at"] = datetime.now(timezone.utc)
        if "content" in update_data:
            title = update_data.get("title") or req.title or "Manual Input"
            update_data["text"] = f"{title}: {update_data['content']}"
            update_data["heading_to_text"] = f"{title} > {update_data['content']}"
            update_data["status"] = "pending"

        query = {"_id": obj_id}
        if self.agent_id:
            query = {"$and": [{"_id": obj_id}, self._get_agent_filter()]}

        self.collection.update_one(query, {"$set": update_data})
        return self.get_rag(rag_id)

    async def delete_rag(self, rag_id: str) -> bool:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")
        
        query = {"_id": obj_id}
        if self.agent_id:
            query = {"$and": [{"_id": obj_id}, self._get_agent_filter()]}
            
        res = self.collection.delete_one(query)
        return res.deleted_count > 0

    def get_rag(self, rag_id: str) -> Optional[RagResponse]:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")
            
        query = {"_id": obj_id}
        if self.agent_id:
            agent_filter = self._get_agent_filter()
            agent_filter["$or"].append({"agent_id": None})
            query = {"$and": [{"_id": obj_id}, agent_filter]}
            
        doc = self.collection.find_one(query)
        if not doc:
            return None
        return format_rag_response(doc)

    def get_all_rags(self) -> List[RagResponse]:
        query = {}
        if self.agent_id:
            agent_filter = self._get_agent_filter()
            agent_filter["$or"].append({"agent_id": None})
            query = agent_filter
            
        docs = self.collection.find(query).sort("updated_at", -1)
        return [format_rag_response(d) for d in docs]

    async def update_embedding(self, rag_id: str):
        try:
            obj_id = ObjectId(rag_id)
            doc = self.collection.find_one({"_id": obj_id})
            if not doc:
                return
            
            emb_text = doc.get("heading_to_text") or doc.get("text")
            emb = await self.vector_service.create_embedding(emb_text)
            
            self.collection.update_one(
                {"_id": obj_id},
                {"$set": {"embedding": emb, "status": "completed", "updated_at": datetime.now(timezone.utc)}}
            )
        except Exception as e:
            print(f"Error updating embedding for {rag_id}: {e}")
            try:
                self.collection.update_one(
                    {"_id": ObjectId(rag_id)},
                    {"$set": {"status": "error", "updated_at": datetime.now(timezone.utc)}}
                )
            except:
                pass

    async def ingest_file_parse(self, file_id: str, org_id: str) -> bool:
        """
        Main entrypoint for parsing ingestion. Reconstructs tree structure,
        generates leaves, embeds them, and saves to file_parsed_flattened.
        """
        if not self.parse_collection:
            return False
            
        parse_doc = self.parse_collection.find_one({"file_id": file_id})
        if not parse_doc:
            return False
            
        result = parse_doc.get("result", {})
        parent_doc_id = parse_doc["_id"] # Use file_parse record's _id as parent_doc_id
        
        # 1. Run the flattener and hierarchy parser
        nested_tree, leaves = NestedDataEnrichmentService.build_hierarchy_and_leaves(
            parse_result=result,
            file_id=file_id,
            org_id=org_id
        )
        
        if not leaves:
            return False
            
        # 2. Bulk embed leaves to save processing time
        texts_to_embed = [leaf["heading_to_text"] for leaf in leaves]
        embeddings = await self.vector_service.bulk_create_embeddings(texts_to_embed)
        
        # 3. Add details and insert leaves
        db_leaves = []
        for idx, leaf in enumerate(leaves):
            leaf["parent_doc_id"] = parent_doc_id
            leaf["embedding"] = embeddings[idx] if idx < len(embeddings) else None
            leaf["status"] = "completed"
            leaf["created_at"] = datetime.now(timezone.utc)
            leaf["updated_at"] = datetime.now(timezone.utc)
            # Retain agent scoping if present in file metadata
            leaf["agent_id"] = self.agent_id
            db_leaves.append(leaf)
            
        # Clear out any existing leaves for this file first to make it idempotent
        self.collection.delete_many({"file_id": file_id})
        
        # Insert all leaves
        self.collection.insert_many(db_leaves)
        return True

def format_rag_response(doc: Dict[str, Any]) -> RagResponse:
    return RagResponse(
        id=str(doc["_id"]),
        title=doc.get("heading_text") or doc.get("json_path") or "Manual Input",
        content=doc.get("content") or doc.get("text") or "",
        status=doc.get("status", "completed"),
        agent_id=doc.get("agent_id"),
        created_at=doc.get("created_at") or datetime.now(timezone.utc),
        updated_at=doc.get("updated_at") or datetime.now(timezone.utc),
        question=doc.get("question"),
        answer=doc.get("answer"),
        original_content=doc.get("text")
    )
