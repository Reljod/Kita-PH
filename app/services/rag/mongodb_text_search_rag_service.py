import re
from typing import List, Optional, Dict, Any
from app.db import TenantCollection

class MongoDBTextSearchRagService:
    def __init__(self, collection: TenantCollection):
        """
        Init with the 'file_parsed_flattened' collection wrapped in a TenantCollection.
        """
        self.collection = collection

    async def text_search(self, query: str, limit: int = 100, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes parallel text search against the flattened leaves.
        Tries MongoDB text index search first, then falls back to clean regex term search.
        """
        # Build base filter query (org_id is injected automatically by TenantCollection)
        agent_filter = {}
        if agent_id:
            from app.models.agent import parse_agent_id
            base_id = parse_agent_id(agent_id)[0]
            agent_filter = {
                "$or": [
                    {"agent_id": agent_id},
                    {"agent_id": base_id},
                    {"agent_id": {"$regex": f"^{base_id}(-v\\d+)?$"}},
                    {"agent_id": None}
                ]
            }

        # Method 1: Try Text Index Search
        try:
            # We match using $text operator
            query_filter = {"$text": {"$search": query}}
            if agent_filter:
                query_filter = {"$and": [query_filter, agent_filter]}
                
            projection = {"score": {"$meta": "textScore"}}
            docs = list(
                self.collection.find(query_filter, projection)
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit)
            )
            if docs:
                for doc in docs:
                    doc["search_score"] = doc.get("score", 0.5)
                return docs
        except Exception as e:
            # Text index might not exist yet, fallback to Regex Match
            pass

        # Method 2: Regex Fallback term search
        # Split query by spaces and clean up words
        words = [w.strip("?,.!:;()\"'") for w in query.split() if len(w) > 2]
        stopwords = {"what", "how", "why", "who", "where", "when", "which", "this", "that", "with", "from", "about", "info"}
        search_words = [w for w in words if w.lower() not in stopwords]
        
        if not search_words:
            search_words = [query]
            
        pattern = "|".join(re.escape(w) for w in search_words)
        
        # We search in 'text' and 'heading_to_text' and 'content' fields
        query_filter = {
            "$or": [
                {"text": {"$regex": pattern, "$options": "i"}},
                {"heading_to_text": {"$regex": pattern, "$options": "i"}},
                {"content": {"$regex": pattern, "$options": "i"}}
            ]
        }
        
        if agent_filter:
            query_filter = {"$and": [query_filter, agent_filter]}
            
        docs = list(self.collection.find(query_filter).limit(limit))
        
        # Assign a mock search score for ranking
        for doc in docs:
            doc["search_score"] = 0.5
            
        return docs
