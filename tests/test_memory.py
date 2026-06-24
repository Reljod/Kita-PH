import unittest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.rag import RagResponse


class TestMemoryRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.memory import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_rag_service = MagicMock()
        from app.dependencies import get_rag_service
        cls.test_app.dependency_overrides[get_rag_service] = lambda: cls.mock_rag_service

        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_create_rag_success(self):
        self.mock_rag_service.add_rag = AsyncMock(return_value=RagResponse(
            id="rag-123", title="Test Doc", content="Sample content",
            status="pending", created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        self.mock_rag_service.update_embedding = AsyncMock()
        response = self.client.post("/memory", json={
            "title": "Test Doc",
            "content": "Sample content",
        })
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["id"], "rag-123")
        self.assertEqual(data["title"], "Test Doc")

    def test_create_rag_validation_error(self):
        response = self.client.post("/memory", json={"title": "", "content": ""})
        self.assertEqual(response.status_code, 422)

    def test_get_all_rags_success(self):
        self.mock_rag_service.get_all_rags.return_value = [
            RagResponse(
                id="rag-123", title="Test Doc", content="Sample",
                status="completed", created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]
        response = self.client.get("/memory")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "rag-123")

    def test_search_memory_success(self):
        self.mock_rag_service.search = AsyncMock(return_value=[
            RagResponse(
                id="rag-123", title="Test Doc", content="Sample",
                status="completed", created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ])
        response = self.client.get("/memory/search", params={"query": "test"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "rag-123")

    def test_search_memory_validation_missing_query(self):
        response = self.client.get("/memory/search")
        self.assertEqual(response.status_code, 422)

    def test_get_rag_success(self):
        self.mock_rag_service.get_rag.return_value = RagResponse(
            id="rag-123", title="Test Doc", content="Sample",
            status="completed", created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.get("/memory/rag-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "rag-123")

    def test_get_rag_not_found(self):
        self.mock_rag_service.get_rag.return_value = None
        response = self.client.get("/memory/rag-999")
        self.assertEqual(response.status_code, 404)

    def test_get_rag_value_error(self):
        self.mock_rag_service.get_rag.side_effect = ValueError("Invalid ID")
        response = self.client.get("/memory/rag-bad")
        self.assertEqual(response.status_code, 400)

    def test_update_rag_success(self):
        self.mock_rag_service.edit_rag = AsyncMock(return_value=RagResponse(
            id="rag-123", title="Updated Doc", content="Updated content",
            status="completed", created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        response = self.client.put("/memory/rag-123", json={"title": "Updated Doc"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Updated Doc")

    def test_update_rag_not_found(self):
        self.mock_rag_service.edit_rag = AsyncMock(return_value=None)
        response = self.client.put("/memory/rag-999", json={"title": "Updated"})
        self.assertEqual(response.status_code, 404)

    def test_update_rag_value_error(self):
        self.mock_rag_service.edit_rag = AsyncMock(side_effect=ValueError("Invalid"))
        response = self.client.put("/memory/rag-bad", json={"title": "Updated"})
        self.assertEqual(response.status_code, 400)

    def test_update_rag_triggers_reembedding(self):
        self.mock_rag_service.edit_rag = AsyncMock(return_value=RagResponse(
            id="rag-123", title="Updated Doc", content="Updated content",
            status="pending", created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        self.mock_rag_service.update_embedding = AsyncMock()
        response = self.client.put("/memory/rag-123", json={"content": "New content"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Updated Doc")

    def test_delete_rag_success(self):
        self.mock_rag_service.delete_rag = AsyncMock(return_value=True)
        response = self.client.delete("/memory/rag-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Memory deleted successfully")

    def test_delete_rag_not_found(self):
        self.mock_rag_service.delete_rag = AsyncMock(return_value=False)
        response = self.client.delete("/memory/rag-999")
        self.assertEqual(response.status_code, 404)

    def test_delete_rag_value_error(self):
        self.mock_rag_service.delete_rag = AsyncMock(side_effect=ValueError("Invalid"))
        response = self.client.delete("/memory/rag-bad")
        self.assertEqual(response.status_code, 400)
