import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.llm import LlmResponse


class TestLlmRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.llm import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_llm_service = MagicMock()
        from app.dependencies import get_llm_service
        cls.test_app.dependency_overrides[get_llm_service] = lambda: cls.mock_llm_service
        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_add_llm_success(self):
        self.mock_llm_service.add_llm.return_value = LlmResponse(
            id="llm-123",
            name="GPT-4",
            model="gpt-4-turbo",
            provider="openrouter",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.post("/llm/", json={
            "name": "GPT-4",
            "model": "gpt-4-turbo",
            "provider": "openrouter",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "llm-123")
        self.assertEqual(data["name"], "GPT-4")

    def test_add_llm_validation_error(self):
        response = self.client.post("/llm/", json={"name": "", "model": "", "provider": ""})
        self.assertEqual(response.status_code, 422)

    def test_list_llms_success(self):
        self.mock_llm_service.list_llms.return_value = [
            LlmResponse(
                id="llm-1", name="GPT-4", model="gpt-4-turbo",
                provider="openrouter",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]
        response = self.client.get("/llm/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "llm-1")

    def test_delete_llm_success(self):
        self.mock_llm_service.delete_llm.return_value = True
        response = self.client.delete("/llm/llm-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "LLM deleted successfully")

    def test_delete_llm_not_found(self):
        self.mock_llm_service.delete_llm.return_value = False
        response = self.client.delete("/llm/llm-999")
        self.assertEqual(response.status_code, 404)

    def test_delete_llm_value_error(self):
        self.mock_llm_service.delete_llm.side_effect = ValueError("Invalid LLM ID")
        response = self.client.delete("/llm/llm-xxx")
        self.assertEqual(response.status_code, 400)
