import asyncio
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class RerankingRagService:
    _reranker = None

    @classmethod
    def get_reranker(cls):
        if cls._reranker is None:
            # Load the BGE-Reranker model
            cls._reranker = CrossEncoder('BAAI/bge-reranker-base')
        return cls._reranker

    async def rerank(self, query: str, candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents against a query using BGE-Reranker.
        """
        if not candidates:
            return []
            
        reranker = self.get_reranker()
        
        # Prepare text representations for the CrossEncoder
        # Candidates can be raw dictionaries (sub-trees) or list of docs
        pairs = []
        for doc in candidates:
            # We can use the formatted context text for reranking
            text_to_score = doc.get("context_text") or doc.get("heading_to_text") or doc.get("text") or str(doc)
            pairs.append((query, text_to_score))
            
        loop = asyncio.get_event_loop()
        rerank_scores = await loop.run_in_executor(None, reranker.predict, pairs)
        
        # If there's only one candidate, predict might return a float scalar rather than a list/array
        if isinstance(rerank_scores, (int, float)):
            rerank_scores = [rerank_scores]
            
        # Combine scores and sort candidates
        scored_candidates = []
        for idx, doc in enumerate(candidates):
            score = float(rerank_scores[idx]) if idx < len(rerank_scores) else -9999.0
            doc["rerank_score"] = score
            scored_candidates.append(doc)
            
        # Sort by rerank score descending
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return scored_candidates[:limit]
