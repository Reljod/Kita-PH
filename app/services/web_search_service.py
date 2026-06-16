import httpx
import os
import logging
from typing import Optional, Any, Dict, Protocol

logger = logging.getLogger(__name__)

class WebSearchService(Protocol):
    async def search(
        self,
        query: str,
        country: str = "us",
        language: str = "en",
        autocorrect: bool = True,
        page: int = 1,
        search_type: str = "search",
        date_range: Optional[str] = None
    ) -> Dict[str, Any]:
        ...

class SerperSearchService(WebSearchService):
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"
        
        # Mapping for intuitive date ranges to Serper's 'tbs' parameter
        self._date_range_map = {
            "past_hour": "qdr:h",
            "past_24_hours": "qdr:d",
            "past_week": "qdr:w",
            "past_month": "qdr:m",
            "past_year": "qdr:y"
        }

    async def search(
        self,
        query: str,
        country: str = "us",
        language: str = "en",
        autocorrect: bool = True,
        page: int = 1,
        search_type: str = "search",
        date_range: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")

        import time
        start_time = time.perf_counter()
        truncated_query = query[:150] + "..." if len(query) > 150 else query

        # Map date_range if it's one of our friendly names
        tbs = self._date_range_map.get(date_range, date_range)

        payload = {
            "q": query,
            "gl": country,
            "hl": language,
            "autocorrect": autocorrect,
            "page": page,
            "type": search_type
        }
        if tbs:
            payload["tbs"] = tbs

        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                duration = time.perf_counter() - start_time
                
                # Extract organic count
                organic_count = len(result.get("organic", [])) if isinstance(result, dict) else 0
                
                logger.info(
                    f"Google Serper search completed: query={truncated_query}, organic_results={organic_count}, duration={duration:.3f}s",
                    extra={
                        "query": truncated_query,
                        "country": country,
                        "language": language,
                        "search_type": search_type,
                        "results_count": organic_count,
                        "duration": duration
                    }
                )
                return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                f"Google Serper search failed: query={truncated_query}, duration={duration:.3f}s: {e}",
                extra={
                    "query": truncated_query,
                    "error": str(e),
                    "duration": duration
                },
                exc_info=True
            )
            raise e
