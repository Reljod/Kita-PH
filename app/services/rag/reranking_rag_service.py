import os
import asyncio
import httpx
from typing import List, Dict, Any


OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_MODEL = "cohere/rerank-v3.5"


class RerankingRagService:

    async def rerank(self, query: str, candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents against a query using
        Cohere Rerank v3.5 via OpenRouter's /api/v1/rerank endpoint.
        """
        if not candidates:
            return []

        # Build document text representations for the reranker
        documents = []
        for doc in candidates:
            text_to_score = (
                doc.get("context_text")
                or doc.get("heading_to_text")
                or doc.get("text")
                or str(doc)
            )
            documents.append(text_to_score)

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": len(candidates),  # Retrieve scores for all; we apply our own limit
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    OPENROUTER_RERANK_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            print(f"Reranking API error: {e}. Returning candidates in original order.")
            return candidates[:limit]

        # data["results"] is a list of {index, relevance_score, document?}
        results = data.get("results", [])

        # Map original index back to the candidate doc and attach the score
        scored_candidates = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", 0.0)
            if idx is not None and idx < len(candidates):
                doc = candidates[idx]
                doc["rerank_score"] = score
                scored_candidates.append(doc)

        # Already sorted by relevance descending by the API, but sort defensively
        scored_candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        return scored_candidates[:limit]
