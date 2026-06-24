import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.tool import ToolResponse
from app.models.agent import AgentResponse


class TestToolRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.tool import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_tool_service = MagicMock()
        cls.mock_agent_service = MagicMock()
        from app.dependencies import get_tool_service, get_agent_service
        cls.test_app.dependency_overrides[get_tool_service] = lambda: cls.mock_tool_service
        cls.test_app.dependency_overrides[get_agent_service] = lambda: cls.mock_agent_service
        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    @patch("app.routes.tool.get_available_tools")
    def test_get_available_tools_success(self, mock_get_available):
        mock_get_available.return_value = [{"name": "web_search", "description": "Search the web"}]
        response = self.client.get("/tool/available")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["name"], "web_search")

    def test_get_tools_success(self):
        self.mock_tool_service.get_tools = AsyncMock(return_value=[{
            "_id": "tool-123",
            "name": "web_search",
            "description": "Search the web",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }])
        response = self.client.get("/tool/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "tool-123")

    def test_register_tool_success(self):
        self.mock_tool_service.register_tool = AsyncMock(return_value=True)
        response = self.client.post("/tool/", json={"name": "web_search"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["message"], "Tool registered successfully")

    def test_register_tool_failure(self):
        self.mock_tool_service.register_tool = AsyncMock(return_value=False)
        response = self.client.post("/tool/", json={"name": "already_registered"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_register_tool_value_error(self):
        self.mock_tool_service.register_tool = AsyncMock(side_effect=ValueError("Tool not supported"))
        response = self.client.post("/tool/", json={"name": "unknown_tool"})
        self.assertEqual(response.status_code, 400)

    def test_get_tool_success(self):
        self.mock_tool_service.get_tool = AsyncMock(return_value={
            "_id": "tool-123",
            "name": "web_search",
            "description": "Search the web",
        })
        response = self.client.get("/tool/tool-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "tool-123")

    def test_get_tool_not_found(self):
        self.mock_tool_service.get_tool = AsyncMock(return_value=None)
        response = self.client.get("/tool/tool-999")
        self.assertEqual(response.status_code, 404)

    def test_get_tool_by_name_success(self):
        self.mock_tool_service.get_tool_by_name = AsyncMock(return_value={
            "_id": "tool-123",
            "name": "web_search",
            "description": "Search the web",
        })
        response = self.client.get("/tool/name/web_search")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "web_search")

    def test_get_tool_by_name_not_found(self):
        self.mock_tool_service.get_tool_by_name = AsyncMock(return_value=None)
        response = self.client.get("/tool/name/unknown")
        self.assertEqual(response.status_code, 404)

    def test_deregister_tool_success(self):
        self.mock_tool_service.deregister_tool = AsyncMock(return_value=True)
        response = self.client.delete("/tool/tool-123")
        self.assertEqual(response.status_code, 200)

    def test_deregister_tool_not_found(self):
        self.mock_tool_service.deregister_tool = AsyncMock(return_value=False)
        response = self.client.delete("/tool/tool-999")
        self.assertEqual(response.status_code, 404)

    def test_get_tool_agents_success(self):
        self.mock_agent_service.get_agents_by_tool.return_value = [
            AgentResponse(
                id="agent-1", base_id="agent-1", version=1,
                name="helper", role="Assistant", goal="Help",
                backstory="Born to help", llm_id="llm-1",
                tools=[], created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]
        response = self.client.get("/tool/tool-123/agents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "agent-1")
