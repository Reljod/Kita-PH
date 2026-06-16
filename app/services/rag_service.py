import asyncio
import os
import httpx
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Protocol
from bson import ObjectId
from app.models.rag import RagCreateRequest, RagUpdateRequest, RagResponse, RagDocument
from app.db import TenantCollection

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "perplexity/pplx-embed-v1-0.6b"
RERANK_MODEL = "cohere/rerank-v3.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_openrouter_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url=OPENROUTER_BASE_URL,
    )


async def _create_embedding(text: str) -> list:
    client = _get_openrouter_client()
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


async def _rerank(query: str, documents: list[str], top_n: int) -> list:
    """
    Calls OpenRouter /api/v1/rerank and returns results sorted by relevance_score desc.
    Falls back to an empty list on error.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/rerank",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": RERANK_MODEL, "query": query, "documents": documents, "top_n": top_n},
            )
            response.raise_for_status()
            return response.json().get("results", [])
    except Exception as e:
        print(f"Reranking API error: {e}")
        return []

class IRagService(Protocol):
    async def add_rag(self, req: RagCreateRequest) -> RagResponse:
        ...
    async def edit_rag(self, rag_id: str, req: RagUpdateRequest, agent_id: Optional[str] = None) -> Optional[RagResponse]:
        ...
    async def delete_rag(self, rag_id: str, agent_id: Optional[str] = None) -> bool:
        ...
    def get_rag(self, rag_id: str, agent_id: Optional[str] = None) -> Optional[RagResponse]:
        ...
    def get_all_rags(self, agent_id: Optional[str] = None) -> List[RagResponse]:
        ...
    async def update_embedding(self, rag_id: str):
        ...
    async def search(self, query: str, limit: int = 5, agent_id: Optional[str] = None) -> List[RagResponse]:
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

    def __init__(self, collection: TenantCollection):
        self.collection = collection

    def _get_agent_filter(self, agent_id: Optional[str]) -> dict:
        """Returns a MongoDB filter query matching this agent's base_id, versioned id, or None."""
        if not agent_id:
            return {}
            
        from app.models.agent import parse_agent_id
        base_id = parse_agent_id(agent_id)[0]
        
        return {
            "$or": [
                {"agent_id": agent_id},
                {"agent_id": base_id},
                {"agent_id": {"$regex": f"^{base_id}(-v\\d+)?$"}}
            ]
        }

    async def add_rag(self, req: RagCreateRequest) -> RagResponse:
        from app.models.agent import parse_agent_id
        raw_agent = req.agent_id
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

    async def edit_rag(self, rag_id: str, req: RagUpdateRequest, agent_id: Optional[str] = None) -> Optional[RagResponse]:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")

        update_data = {k: v for k, v in req.model_dump().items() if v is not None}
        if not update_data:
            return self.get_rag(rag_id, agent_id=agent_id)

        update_data["updated_at"] = datetime.now(timezone.utc)
        if "content" in update_data:
            update_data["original_content"] = update_data["content"]
            update_data["status"] = "pending"

        query = {"_id": obj_id}
        if agent_id:
            query = {"$and": [{"_id": obj_id}, self._get_agent_filter(agent_id)]}

        self.collection.update_one(
            query,
            {"$set": update_data}
        )
        
        return self.get_rag(rag_id, agent_id=agent_id)

    async def delete_rag(self, rag_id: str, agent_id: Optional[str] = None) -> bool:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")
        
        query = {"_id": obj_id}
        if agent_id:
            query = {"$and": [{"_id": obj_id}, self._get_agent_filter(agent_id)]}
            
        res = self.collection.delete_one(query)
        return res.deleted_count > 0

    def get_rag(self, rag_id: str, agent_id: Optional[str] = None) -> Optional[RagResponse]:
        try:
            obj_id = ObjectId(rag_id)
        except Exception:
            raise ValueError("Invalid RAG ID")
            
        query = {"_id": obj_id}
        if agent_id:
            agent_filter = self._get_agent_filter(agent_id)
            agent_filter["$or"].append({"agent_id": None})
            query = {"$and": [{"_id": obj_id}, agent_filter]}
            
        doc = self.collection.find_one(query)
        if not doc:
            return None
            
        return format_rag_response(doc)

    def get_all_rags(self, agent_id: Optional[str] = None) -> List[RagResponse]:
        query = {}
        if agent_id:
            agent_filter = self._get_agent_filter(agent_id)
            agent_filter["$or"].append({"agent_id": None})
            query = agent_filter
            
        docs = self.collection.find(query).sort("updated_at", -1)
        return [format_rag_response(d) for d in docs]

    async def update_embedding(self, rag_id: str):
        try:
            from app.services.rag_enrichment_service import RagEnrichmentService
            enricher = RagEnrichmentService(self.collection)
            await enricher.enrich_and_embed(rag_id)
            logger.info(f"Successfully updated RAG embedding: rag_id={rag_id}", extra={"rag_id": rag_id})
        except Exception as e:
            logger.error(f"Error updating embedding for rag_id={rag_id}: {e}", exc_info=True)
            try:
                self.collection.update_one(
                    {"_id": ObjectId(rag_id)},
                    {"$set": {"status": "error", "updated_at": datetime.now(timezone.utc)}}
                )
            except:
                pass

    async def search(self, query: str, limit: int = 5, agent_id: Optional[str] = None) -> List[RagResponse]:
        import time
        start_time = time.perf_counter()
        truncated_query = query[:150] + "..." if len(query) > 150 else query

        # Embed the query via OpenRouter
        try:
            query_embedding = await _create_embedding(query)
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}", exc_info=True)
            return []

        # Atlas Vector Search
        # Retrieve limit * 4 candidates for rerank
        candidate_limit = limit * 4
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "rag_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": candidate_limit * 10,
                    "limit": candidate_limit
                }
            }
        ]

        docs = []
        try:
            # If we have an agent_id, we need to filter results by it or org-wide
            if agent_id:
                agent_filter = self._get_agent_filter(agent_id)
                agent_filter["$or"].append({"agent_id": None})
                pipeline.append({"$match": agent_filter})

            pipeline.append({"$set": {"search_score": {"$meta": "vectorSearchScore"}}})
            docs = list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Error during vector search: {e}", exc_info=True)
            docs = []

        # Fallback to keyword regex-based text search if vector index returned no results
        if not docs:
            import re
            words = [w.strip("?,.!:;()\"'") for w in query.split() if len(w) > 2]
            stopwords = {"what", "how", "why", "who", "where", "when", "which", "this", "that", "with", "from", "about", "info"}
            search_words = [w for w in words if w.lower() not in stopwords] or [query]
            pattern = "|".join(re.escape(w) for w in search_words)
            query_filter: dict = {
                "$or": [
                    {"title": {"$regex": pattern, "$options": "i"}},
                    {"content": {"$regex": pattern, "$options": "i"}},
                    {"question": {"$regex": pattern, "$options": "i"}},
                ]
            }
            if agent_id:
                agent_filter = self._get_agent_filter(agent_id)
                agent_filter["$or"].append({"agent_id": None})
                query_filter = {"$and": [query_filter, agent_filter]}
            docs = list(self.collection.find(query_filter).sort("updated_at", -1).limit(candidate_limit))

        if not docs:
            duration = time.perf_counter() - start_time
            logger.info(
                f"RAG search completed: query={truncated_query}, results=0, duration={duration:.3f}s",
                extra={
                    "query": truncated_query,
                    "limit": limit,
                    "agent_id": agent_id,
                    "results_count": 0,
                    "duration": duration
                }
            )
            return []

        # Rerank candidates via OpenRouter cohere/rerank-v3.5
        rerank_results = []
        doc_texts = [doc.get("question") or f"{doc.get('title', '')}: {doc.get('content', '')}" for doc in docs]
        try:
            rerank_results = await _rerank(query, doc_texts, top_n=len(docs))
        except Exception as re_err:
            logger.error(f"Failed to rerank: {re_err}", exc_info=True)

        # Build an index→score map from rerank results
        rerank_score_map: dict[int, float] = {}
        for item in rerank_results:
            rerank_score_map[item["index"]] = item.get("relevance_score", 0.5)

        # Compute combined score: rerank relevance (or vector search score) + recency boost
        now = datetime.now(timezone.utc)
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

            # Baseline: use cohere rerank score if available, else fall back to vector search score
            score = rerank_score_map.get(idx, doc.get("search_score", 0.5))

            # Combine baseline score + recency boost
            doc["combined_score"] = score + 0.2 * recency_score

        # Sort and take top limit
        docs.sort(key=lambda d: d.get("combined_score", 0.0), reverse=True)
        docs = docs[:limit]

        duration = time.perf_counter() - start_time
        logger.info(
            f"RAG search completed: query={truncated_query}, results={len(docs)}, duration={duration:.3f}s",
            extra={
                "query": truncated_query,
                "limit": limit,
                "agent_id": agent_id,
                "results_count": len(docs),
                "duration": duration
            }
        )

        return [format_rag_response(d) for d in docs]
