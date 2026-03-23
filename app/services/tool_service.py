from typing import Optional, Any, Dict, Protocol
from app.services.web_search_service import WebSearchService

class IToolService(Protocol):
    async def web_search(
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

class ToolService(IToolService):
    def __init__(self, web_search_service: WebSearchService):
        self.web_search_service = web_search_service

    async def web_search(
        self,
        query: str,
        country: str = "us",
        language: str = "en",
        autocorrect: bool = True,
        page: int = 1,
        search_type: str = "search",
        date_range: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self.web_search_service.search(
            query=query,
            country=country,
            language=language,
            autocorrect=autocorrect,
            page=page,
            search_type=search_type,
            date_range=date_range
        )
