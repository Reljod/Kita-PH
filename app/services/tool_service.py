from typing import Optional, Any, Dict, Protocol, List
from app.services.web_search_service import WebSearchService
from app.services.tools import get_available_tools
from app.db import TenantCollection
from datetime import datetime, timezone
from app.exceptions import (
    SystemConfigurationError,
    ToolNotFoundError,
    ToolRegistrationError
)

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

    async def register_tool(self, name: str) -> bool:
        ...

    async def deregister_tool(self, tool_id: str) -> bool:
        ...

    async def get_tools(self) -> List[dict]:
        ...

    async def get_tool(self, tool_id: str) -> Optional[dict]:
        ...

    async def get_tool_by_name(self, name: str) -> Optional[dict]:
        ...

class ToolService(IToolService):
    def __init__(self, web_search_service: WebSearchService, collection: Optional[TenantCollection] = None):
        self.web_search_service = web_search_service
        self.collection = collection

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

    async def register_tool(self, name: str) -> bool:
        if not self.collection:
            raise SystemConfigurationError("Collection not provided for ToolService")

        available_tools = get_available_tools()
        if name not in available_tools:
            raise ToolRegistrationError(name, "Tool not found in available tools")

        description = available_tools[name]
        
        # Check if already registered
        existing = self.collection.find_one({"name": name})
        if existing:
            return True

        self.collection.insert_one({
            "name": name,
            "description": description,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        })
        return True

    async def deregister_tool(self, tool_id: str) -> bool:
        if not self.collection:
            raise SystemConfigurationError("Collection not provided for ToolService")
        
        from bson import ObjectId
        try:
            obj_id = ObjectId(tool_id)
        except Exception as e:
            raise ToolRegistrationError(tool_id, f"Invalid tool ID format: {str(e)}")

        res = self.collection.delete_one({"_id": obj_id})
        if res.deleted_count == 0:
            raise ToolNotFoundError(tool_id)
        return True

    async def get_tools(self) -> List[dict]:
        if not self.collection:
            raise SystemConfigurationError("Collection not provided for ToolService")
        return list(self.collection.find({}))

    async def get_tool(self, tool_id: str) -> dict:
        if not self.collection:
            raise SystemConfigurationError("Collection not provided for ToolService")
        from bson import ObjectId
        try:
            obj_id = ObjectId(tool_id)
        except Exception as e:
            raise ToolNotFoundError(tool_id, message=f"Invalid tool ID format: {str(e)}")

        tool = self.collection.find_one({"_id": obj_id})
        if not tool:
            raise ToolNotFoundError(tool_id)
        return tool

    async def get_tool_by_name(self, name: str) -> dict:
        if not self.collection:
            raise SystemConfigurationError("Collection not provided for ToolService")
        tool = self.collection.find_one({"name": name})
        if not tool:
            raise ToolNotFoundError(name)
        return tool
