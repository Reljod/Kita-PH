import unittest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.agent import AgentResponse
from app.models.chat import ChatResponse
from app.models.rag import RagResponse


class TestAgentRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.agent import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_agent_service = MagicMock()
        cls.mock_chat_service = MagicMock()
        cls.mock_rag_service = MagicMock()

        from app.dependencies import get_agent_service, get_chat_service, get_agent_rag_service
        cls.test_app.dependency_overrides[get_agent_service] = lambda: cls.mock_agent_service
        cls.test_app.dependency_overrides[get_chat_service] = lambda: cls.mock_chat_service
        cls.test_app.dependency_overrides[get_agent_rag_service] = lambda: cls.mock_rag_service
        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def _agent_response(self, **kwargs) -> AgentResponse:
        fields = dict(
            id="agent-123", base_id="agent-123", version=1,
            name="Helper", role="Assistant", goal="Help users",
            backstory="Born to help", llm_id="llm-1", tools=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        fields.update(kwargs)
        return AgentResponse(**fields)

    def _chat_response(self, **kwargs) -> ChatResponse:
        fields = dict(
            id="chat-123", messages=[], agent_id="agent-123",
            preview=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        fields.update(kwargs)
        return ChatResponse(**fields)

    def _rag_response(self, **kwargs) -> RagResponse:
        fields = dict(
            id="rag-123", title="Doc", content="Content",
            status="completed",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        fields.update(kwargs)
        return RagResponse(**fields)

    def test_create_agent_success(self):
        self.mock_agent_service.create_agent = AsyncMock(return_value=self._agent_response())
        response = self.client.post("/agent/", json={
            "name": "Helper", "role": "Assistant",
            "goal": "Help users", "backstory": "Born to help",
            "llm_id": "llm-1",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "agent-123")

    def test_create_agent_value_error(self):
        self.mock_agent_service.create_agent = AsyncMock(side_effect=ValueError("Invalid agent data"))
        response = self.client.post("/agent/", json={
            "name": "Helper", "role": "Assistant",
            "goal": "Help users", "backstory": "Born to help",
            "llm_id": "llm-1",
        })
        self.assertEqual(response.status_code, 400)

    def test_create_agent_validation_error(self):
        response = self.client.post("/agent/", json={
            "name": "", "role": "", "goal": "",
            "backstory": "", "llm_id": "",
        })
        self.assertEqual(response.status_code, 422)

    def test_update_agent_success(self):
        self.mock_agent_service.update_agent = AsyncMock(return_value=self._agent_response(name="Updated Helper"))
        response = self.client.put("/agent/agent-123", json={"name": "Updated Helper"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "Updated Helper")

    def test_update_agent_not_found(self):
        self.mock_agent_service.update_agent = AsyncMock(return_value=None)
        response = self.client.put("/agent/agent-999", json={"name": "Updated"})
        self.assertEqual(response.status_code, 404)

    def test_get_agent_success(self):
        self.mock_agent_service.get_agent.return_value = self._agent_response()
        response = self.client.get("/agent/agent-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "agent-123")

    def test_get_agent_not_found(self):
        self.mock_agent_service.get_agent.return_value = None
        response = self.client.get("/agent/agent-999")
        self.assertEqual(response.status_code, 404)

    def test_get_all_agents_success(self):
        self.mock_agent_service.get_all_agents.return_value = [self._agent_response()]
        response = self.client.get("/agent/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "agent-123")

    def test_delete_agent_success(self):
        self.mock_agent_service.delete_agent.return_value = True
        response = self.client.delete("/agent/agent-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Agent deleted successfully")

    def test_delete_agent_not_found(self):
        self.mock_agent_service.delete_agent.return_value = False
        response = self.client.delete("/agent/agent-999")
        self.assertEqual(response.status_code, 404)

    def test_create_agent_chat_sse(self):
        async def mock_stream(*args, **kwargs):
            yield {"type": "text", "content": "Hello"}
            yield {"type": "done", "content": ""}
        self.mock_chat_service.create_chat_stream = mock_stream
        response = self.client.post("/agent/agent-123/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        self.assertIn("data:", response.text)

    def test_create_agent_chat_sse_error(self):
        async def mock_stream_error(*args, **kwargs):
            raise Exception("Chat error")
        self.mock_chat_service.create_chat_stream = mock_stream_error
        response = self.client.post("/agent/agent-123/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        content = response.text
        self.assertIn("error", content)

    def test_continue_agent_chat_sse(self):
        async def mock_continue(*args, **kwargs):
            yield {"type": "text", "content": "Continuing"}
            yield {"type": "done", "content": ""}
        self.mock_chat_service.continue_chat_stream = mock_continue
        response = self.client.post("/agent/agent-123/chat/chat-456/continue", json={"message": "Continue"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

    def test_get_agent_chat_success(self):
        self.mock_chat_service.get_chat.return_value = self._chat_response()
        response = self.client.get("/agent/agent-123/chat/chat-456")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "chat-123")

    def test_get_agent_chat_not_found(self):
        self.mock_chat_service.get_chat.return_value = None
        response = self.client.get("/agent/agent-123/chat/chat-999")
        self.assertEqual(response.status_code, 404)

    def test_get_agent_chat_value_error(self):
        self.mock_chat_service.get_chat.side_effect = ValueError("Invalid ID")
        response = self.client.get("/agent/agent-123/chat/chat-bad")
        self.assertEqual(response.status_code, 400)

    def test_get_all_agent_chats_success(self):
        self.mock_chat_service.get_all_chats.return_value = [self._chat_response()]
        response = self.client.get("/agent/agent-123/chat")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "chat-123")

    def test_create_agent_rag_success(self):
        self.mock_rag_service.add_rag = AsyncMock(return_value=self._rag_response())
        self.mock_rag_service.update_embedding = AsyncMock()
        response = self.client.post("/agent/agent-123/memory", json={
            "title": "Doc",
            "content": "Content",
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["id"], "rag-123")

    def test_create_agent_rag_error(self):
        self.mock_rag_service.add_rag = AsyncMock(side_effect=Exception("DB error"))
        response = self.client.post("/agent/agent-123/memory", json={
            "title": "Doc",
            "content": "Content",
        })
        self.assertEqual(response.status_code, 500)

    def test_get_all_agent_rags_success(self):
        self.mock_rag_service.get_all_rags.return_value = [self._rag_response()]
        response = self.client.get("/agent/agent-123/memory")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "rag-123")

    def test_search_agent_memory_success(self):
        self.mock_rag_service.search = AsyncMock(return_value=[self._rag_response()])
        response = self.client.get("/agent/agent-123/memory/search", params={"query": "test"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "rag-123")

    def test_search_agent_memory_missing_query(self):
        response = self.client.get("/agent/agent-123/memory/search")
        self.assertEqual(response.status_code, 422)

    def test_get_agent_rag_success(self):
        self.mock_rag_service.get_rag.return_value = self._rag_response()
        response = self.client.get("/agent/agent-123/memory/rag-456")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "rag-123")

    def test_get_agent_rag_not_found(self):
        self.mock_rag_service.get_rag.return_value = None
        response = self.client.get("/agent/agent-123/memory/rag-999")
        self.assertEqual(response.status_code, 404)

    def test_get_agent_rag_value_error(self):
        self.mock_rag_service.get_rag.side_effect = ValueError("Invalid ID")
        response = self.client.get("/agent/agent-123/memory/rag-bad")
        self.assertEqual(response.status_code, 400)

    def test_update_agent_rag_success(self):
        self.mock_rag_service.edit_rag = AsyncMock(return_value=self._rag_response(title="Updated"))
        response = self.client.put("/agent/agent-123/memory/rag-456", json={"title": "Updated"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Updated")

    def test_update_agent_rag_not_found(self):
        self.mock_rag_service.edit_rag = AsyncMock(return_value=None)
        response = self.client.put("/agent/agent-123/memory/rag-999", json={"title": "Updated"})
        self.assertEqual(response.status_code, 404)

    def test_update_agent_rag_value_error(self):
        self.mock_rag_service.edit_rag = AsyncMock(side_effect=ValueError("Invalid"))
        response = self.client.put("/agent/agent-123/memory/rag-bad", json={"title": "Updated"})
        self.assertEqual(response.status_code, 400)

    def test_update_agent_rag_triggers_reembedding(self):
        self.mock_rag_service.edit_rag = AsyncMock(return_value=self._rag_response(status="pending"))
        self.mock_rag_service.update_embedding = AsyncMock()
        response = self.client.put("/agent/agent-123/memory/rag-456", json={"content": "New content"})
        self.assertEqual(response.status_code, 200)

    def test_delete_agent_rag_success(self):
        self.mock_rag_service.delete_rag = AsyncMock(return_value=True)
        response = self.client.delete("/agent/agent-123/memory/rag-456")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Memory deleted successfully")

    def test_delete_agent_rag_not_found(self):
        self.mock_rag_service.delete_rag = AsyncMock(return_value=False)
        response = self.client.delete("/agent/agent-123/memory/rag-999")
        self.assertEqual(response.status_code, 404)

    def test_delete_agent_rag_value_error(self):
        self.mock_rag_service.delete_rag = AsyncMock(side_effect=ValueError("Invalid"))
        response = self.client.delete("/agent/agent-123/memory/rag-bad")
        self.assertEqual(response.status_code, 400)

    def test_add_agent_tools_success(self):
        self.mock_agent_service.add_tools = AsyncMock(return_value=True)
        self.mock_agent_service.get_agent.return_value = self._agent_response(tools=["tool-1"])
        response = self.client.post("/agent/agent-123/tools/add", json={"tool_ids": ["tool-1"]})
        self.assertEqual(response.status_code, 200)
        self.assertIn("tools", response.json())

    def test_add_agent_tools_agent_not_found(self):
        self.mock_agent_service.add_tools = AsyncMock(return_value=False)
        response = self.client.post("/agent/agent-123/tools/add", json={"tool_ids": ["tool-1"]})
        self.assertEqual(response.status_code, 404)

    def test_remove_agent_tools_success(self):
        self.mock_agent_service.remove_tools = AsyncMock(return_value=True)
        self.mock_agent_service.get_agent.return_value = self._agent_response(tools=[])
        response = self.client.post("/agent/agent-123/tools/remove", json={"tool_ids": ["tool-1"]})
        self.assertEqual(response.status_code, 200)

    def test_remove_agent_tools_agent_not_found(self):
        self.mock_agent_service.remove_tools = AsyncMock(return_value=False)
        response = self.client.post("/agent/agent-123/tools/remove", json={"tool_ids": ["tool-1"]})
        self.assertEqual(response.status_code, 404)
