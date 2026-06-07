import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from bson import ObjectId
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
        updated_at=doc["updated_at"],
        question=doc.get("question"),
        answer=doc.get("answer"),
        original_content=doc.get("original_content")
    )

class MongoVectorDbRagService(IRagService):
    _model = None
    _reranker = None

    def __init__(self, collection: TenantCollection, agent_id: Optional[str] = None):
        self.collection = collection
        self.agent_id = agent_id

    def _get_agent_filter(self) -> dict:
        """Returns a MongoDB filter query matching this agent's base_id, versioned id, or None."""
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

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # We initialize the model lazily
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    @classmethod
    def get_reranker(cls):
        if cls._reranker is None:
            from sentence_transformers import CrossEncoder
            cls._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return cls._reranker

    async def add_rag(self, req: RagCreateRequest) -> RagResponse:
        from app.models.agent import parse_agent_id
        raw_agent = req.agent_id or self.agent_id
        agent_id = parse_agent_id(raw_agent)[0] if raw_agent else None

        new_rag = RagDocument(
            title=req.title,
            content=req.content,
            original_content=req.content,
            agent_id=agent_id,
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
        if "content" in update_data:
            update_data["original_content"] = update_data["content"]
            update_data["status"] = "pending"

        query = {"_id": obj_id}
        if self.agent_id:
            query = {"$and": [{"_id": obj_id}, self._get_agent_filter()]}

        self.collection.update_one(
            query,
            {"$set": update_data}
        )
        
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
            from app.services.rag_enrichment_service import RagEnrichmentService
            enricher = RagEnrichmentService(self.collection)
            await enricher.enrich_and_embed(rag_id, self.get_model())
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
        # Retrieve limit * 4 candidates for rerank
        candidate_limit = limit * 4
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "rag_index",
                    "path": "embedding",
                    "queryVector": query_embedding.tolist(),
                    "numCandidates": candidate_limit * 10,
                    "limit": candidate_limit
                }
            }
        ]
        
        docs = []
        try:
            # If we have an agent_id, we need to filter results by it or org-wide
            if self.agent_id:
                agent_filter = self._get_agent_filter()
                agent_filter["$or"].append({"agent_id": None})
                pipeline.append({
                    "$match": agent_filter
                })
            
            pipeline.append({
                "$set": {
                    "search_score": {"$meta": "vectorSearchScore"}
                }
            })
            
            docs = list(self.collection.aggregate(pipeline))
        except Exception as e:
            print(f"Error during vector search: {e}")
            docs = []

        # Fallback to keyword regex-based text search if vector index is not set up or returned no results
        if not docs:
            # Split query by spaces and clean up words
            import re
            words = [w.strip("?,.!:;()\"'") for w in query.split() if len(w) > 2]
            stopwords = {"what", "how", "why", "who", "where", "when", "which", "this", "that", "with", "from", "about", "info"}
            search_words = [w for w in words if w.lower() not in stopwords]
            
            if not search_words:
                search_words = [query]
                
            pattern = "|".join(re.escape(w) for w in search_words)
            query_filter = {
                "$or": [
                    {"title": {"$regex": pattern, "$options": "i"}},
                    {"content": {"$regex": pattern, "$options": "i"}},
                    {"question": {"$regex": pattern, "$options": "i"}}
                ]
            }
            if self.agent_id:
                agent_filter = self._get_agent_filter()
                agent_filter["$or"].append({"agent_id": None})
                query_filter = {
                    "$and": [
                        query_filter,
                        agent_filter
                    ]
                }
            docs = list(self.collection.find(query_filter).sort("updated_at", -1).limit(candidate_limit))

        if not docs:
            return []

        # Rerank candidates using CrossEncoder if available
        rerank_scores = None
        try:
            reranker = self.get_reranker()
            pairs = [(query, doc.get("question") or f"{doc['title']}: {doc['content']}") for doc in docs]
            rerank_scores = await loop.run_in_executor(None, reranker.predict, pairs)
            
            # If we only have one result, rerank_scores might be a float scalar rather than a list/array
            if isinstance(rerank_scores, (int, float)):
                rerank_scores = [rerank_scores]
        except Exception as re_err:
            print(f"Failed to use cross-encoder for reranking: {re_err}")

        # Compute combined score: baseline score (cross-encoder or search_score) + recency boost
        now = datetime.utcnow()
        for idx, doc in enumerate(docs):
            # Recency score calculation
            created_at = doc.get("created_at") or now
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except:
                    created_at = now
            
            age_in_days = (now.replace(tzinfo=None) - created_at.replace(tzinfo=None)).total_seconds() / 86400.0
            age_in_days = max(0.0, age_in_days)
            # Recency decays from 1.0 (newest) to near 0.0 (old) with a half-life of ~30 days
            recency_score = 1.0 / (1.0 + age_in_days / 30.0)

            # Determine baseline score (from reranker or search score or default)
            if rerank_scores is not None and idx < len(rerank_scores):
                import math
                try:
                    # Convert logit output to a normalized 0-1 range using sigmoid
                    score = 1.0 / (1.0 + math.exp(-float(rerank_scores[idx])))
                except:
                    score = 0.5
            else:
                score = doc.get("search_score", 0.5)

            # Combine baseline score (weight 0.8) and recency score (weight 0.2)
            doc["combined_score"] = score + 0.2 * recency_score

        # Sort and take top limit
        docs.sort(key=lambda d: d.get("combined_score", 0.0), reverse=True)
        docs = docs[:limit]

        return [format_rag_response(d) for d in docs]
