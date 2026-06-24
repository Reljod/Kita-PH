import unittest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.chat import ChatResponse


class TestChatRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.chat import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_chat_service = MagicMock()

        from app.dependencies import get_chat_service
        from app.security import require_org_membership

        cls.test_app.dependency_overrides[get_chat_service] = lambda: cls.mock_chat_service
        cls.test_app.dependency_overrides[require_org_membership] = lambda: "org-123"

        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_create_chat_sse(self):
        async def mock_stream(*args, **kwargs):
            yield {"type": "text", "content": "Hello"}
            yield {"type": "done", "content": ""}

        self.mock_chat_service.create_chat_stream = mock_stream

        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        content = response.text
        self.assertIn("data:", content)

    def test_create_chat_sse_error_handling(self):
        async def mock_stream_error(*args, **kwargs):
            raise Exception("Something went wrong")
            yield  # pragma: no cover

        self.mock_chat_service.create_chat_stream = mock_stream_error

        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        content = response.text
        self.assertIn("error", content)

    def test_continue_chat_sse(self):
        async def mock_continue_stream(*args, **kwargs):
            yield {"type": "text", "content": "Continuing"}
            yield {"type": "done", "content": ""}

        self.mock_chat_service.continue_chat_stream = mock_continue_stream

        response = self.client.post("/chat/chat-123/continue", json={"message": "Continue"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

    def test_get_chat_success(self):
        self.mock_chat_service.get_chat.return_value = ChatResponse(
            id="chat-123",
            messages=[{"role": "user", "content": "Hello"}],
            agent_id=None,
            preview=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.get("/chat/chat-123")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "chat-123")

    def test_get_chat_not_found(self):
        self.mock_chat_service.get_chat.return_value = None
        response = self.client.get("/chat/chat-999")
        self.assertEqual(response.status_code, 404)

    def test_get_chat_value_error(self):
        self.mock_chat_service.get_chat.side_effect = ValueError("Invalid chat ID")
        response = self.client.get("/chat/chat-bad")
        self.assertEqual(response.status_code, 400)

    def test_get_all_chats_success(self):
        self.mock_chat_service.get_all_chats.return_value = [
            ChatResponse(
                id="chat-123",
                messages=[],
                agent_id=None,
                preview=None,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]
        response = self.client.get("/chat")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "chat-123")

    def test_get_all_chats_empty(self):
        self.mock_chat_service.get_all_chats.return_value = []
        response = self.client.get("/chat")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    @patch("app.dependencies.services.get_services")
    def test_get_agent_status_success(self, mock_get_services):
        mock_status_service = MagicMock()
        mock_status_service.get_status = AsyncMock(return_value={
            "status": "completed",
            "progress": 100,
        })
        mock_get_services.return_value.agent_status_service = mock_status_service

        response = self.client.get("/chat/status/status-key-123")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["progress"], 100)

    @patch("app.dependencies.services.get_services")
    def test_get_agent_status_not_found(self, mock_get_services):
        mock_status_service = MagicMock()
        mock_status_service.get_status = AsyncMock(return_value=None)
        mock_get_services.return_value.agent_status_service = mock_status_service

        response = self.client.get("/chat/status/status-key-999")
        self.assertEqual(response.status_code, 404)

    def test_create_chat_validation_error(self):
        response = self.client.post("/chat", json={"message": ""})
        self.assertEqual(response.status_code, 422)

    def test_continue_chat_validation_error(self):
        response = self.client.post("/chat/chat-123/continue", json={"message": ""})
        self.assertEqual(response.status_code, 422)
