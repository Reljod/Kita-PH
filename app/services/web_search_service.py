import httpx
import os
from typing import Optional, Any, Dict, Protocol

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

        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
