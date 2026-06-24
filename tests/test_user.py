import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.user import UserResponse


class TestUserRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.user import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_user_service = MagicMock()
        from app.security import get_current_user, get_user_service, oauth2_scheme

        cls.current_user = UserResponse(
            id="user-123",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        cls.test_app.dependency_overrides[get_current_user] = lambda: cls.current_user
        cls.test_app.dependency_overrides[get_user_service] = lambda: cls.mock_user_service
        cls.test_app.dependency_overrides[oauth2_scheme] = lambda: "test-token"

        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_get_me_success(self):
        response = self.client.get("/user/me")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "user-123")
        self.assertEqual(data["email"], "test@example.com")

    def test_update_me_success(self):
        updated = UserResponse(
            id="user-123",
            email="test@example.com",
            first_name="Jane",
            last_name="Smith",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.mock_user_service.update_user.return_value = updated
        response = self.client.patch("/user/me", json={"first_name": "Jane", "last_name": "Smith"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["first_name"], "Jane")
        self.assertEqual(data["last_name"], "Smith")

    def test_update_me_validation_error(self):
        response = self.client.patch("/user/me", json={"first_name": ""})
        self.assertEqual(response.status_code, 422)

    def test_update_password_success(self):
        self.mock_user_service.update_password.return_value = True
        response = self.client.patch("/user/me/password", json={
            "old_password": "oldpass123",
            "new_password": "newpass456",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Password updated successfully")

    def test_update_password_invalid_current(self):
        self.mock_user_service.update_password.return_value = False
        response = self.client.patch("/user/me/password", json={
            "old_password": "wrongpass",
            "new_password": "newpass456",
        })
        self.assertEqual(response.status_code, 400)

    def test_update_password_validation_error(self):
        response = self.client.patch("/user/me/password", json={
            "old_password": "short",
            "new_password": "new",
        })
        self.assertEqual(response.status_code, 422)
