import os
import asyncio
from typing import List, Optional, Dict, Any

from app.db import TenantCollection


def _get_openrouter_client():
    """Returns an openai.AsyncOpenAI client pointed at OpenRouter's embeddings endpoint."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
    )


class MongoDBVectorSearchRagService:
    _client = None

    EMBEDDING_MODEL = "perplexity/pplx-embed-v1-0.6b"

    def __init__(self, collection: TenantCollection):
        """
        Init with the 'file_parsed_flattened' collection wrapped in a TenantCollection.
        """
        self.collection = collection

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = _get_openrouter_client()
        return cls._client

    async def create_embedding(self, text: str) -> List[float]:
        client = self.get_client()
        response = await client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    async def bulk_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self.get_client()
        response = await client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=texts,
        )
        # Results are returned in the same order as the input
        return [item.embedding for item in response.data]

    async def vector_search(self, query: str, limit: int = 100, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Finds the top leaf nodes using Atlas Vector Search.
        """
        query_embedding = await self.create_embedding(query)

        # Atlas Vector Search Stage
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "knowledge_base_rag", # default Atlas vector index name on file_parsed_flattened
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            }
        ]

        # Build filter conditions
        # In TenantCollection, org_id filter is injected automatically.
        # We can also add agent_id scoping if requested.
        agent_filter = {}
        if agent_id:
            from app.models.agent import parse_agent_id
            base_id = parse_agent_id(agent_id)[0]
            agent_filter = {
                "$or": [
                    {"agent_id": agent_id},
                    {"agent_id": base_id},
                    {"agent_id": {"$regex": f"^{base_id}(-v\\d+)?$"}},
                    {"agent_id": None} # Also allow org-wide items
                ]
            }
            pipeline.append({
                "$match": agent_filter
            })

        pipeline.append({
            "$set": {
                "search_score": {"$meta": "vectorSearchScore"}
            }
        })

        try:
            docs = list(self.collection.aggregate(pipeline))
            return docs
        except Exception as e:
            print(f"Error during vector search aggregate: {e}")
            # Safe fallback: return empty list to fallback to text search
            return []
