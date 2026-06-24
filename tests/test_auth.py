import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.auth import Token


class TestAuthRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.auth import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_auth_service = MagicMock()
        cls.mock_user_service = MagicMock()
        cls.mock_org_service = MagicMock()

        from app.security import get_auth_service, get_user_service, get_org_service, oauth2_scheme

        cls.test_app.dependency_overrides[get_auth_service] = lambda: cls.mock_auth_service
        cls.test_app.dependency_overrides[get_user_service] = lambda: cls.mock_user_service
        cls.test_app.dependency_overrides[get_org_service] = lambda: cls.mock_org_service
        cls.test_app.dependency_overrides[oauth2_scheme] = lambda: "test-token"

        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_register_success(self):
        self.mock_user_service.get_user_by_email.return_value = None
        self.mock_user_service.hash_password_async = AsyncMock(return_value="hashed_pass")
        mock_user = MagicMock()
        mock_user.id = "user-123"
        self.mock_user_service.create_user.return_value = mock_user
        self.mock_auth_service.generate_tokens.return_value = Token(
            access_token="access-token",
            refresh_token="refresh-token",
            token_type="bearer",
        )

        response = self.client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "securepassword",
            "first_name": "John",
            "last_name": "Doe",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["access_token"], "access-token")
        self.assertEqual(data["refresh_token"], "refresh-token")
        self.assertEqual(data["token_type"], "bearer")

    def test_register_duplicate_email(self):
        self.mock_user_service.get_user_by_email.return_value = {"_id": "existing-user"}

        response = self.client.post("/auth/register", json={
            "email": "existing@example.com",
            "password": "securepassword",
            "first_name": "John",
            "last_name": "Doe",
        })
        self.assertEqual(response.status_code, 400)

    def test_register_validation_error(self):
        response = self.client.post("/auth/register", json={
            "email": "not-an-email",
            "password": "123",
            "first_name": "",
            "last_name": "",
        })
        self.assertEqual(response.status_code, 422)

    def test_login_success_no_org(self):
        self.mock_user_service.get_user_by_email.return_value = {
            "_id": "user-123",
            "password": "hashed_pass",
        }
        self.mock_user_service.verify_password_async = AsyncMock(return_value=True)
        self.mock_org_service.get_user_orgs.return_value = []
        self.mock_auth_service.generate_tokens.return_value = Token(
            access_token="access-token",
            refresh_token="refresh-token",
            token_type="bearer",
        )

        response = self.client.post("/auth/login", data={
            "username": "test@example.com",
            "password": "securepassword",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["access_token"], "access-token")
        self.assertEqual(data["token_type"], "bearer")

    def test_login_success_with_org(self):
        self.mock_user_service.get_user_by_email.return_value = {
            "_id": "user-123",
            "password": "hashed_pass",
        }
        self.mock_user_service.verify_password_async = AsyncMock(return_value=True)
        self.mock_org_service.get_user_orgs.return_value = [
            MagicMock(id="org-123", spec=[])
        ]
        mock_org_member = MagicMock()
        mock_org_member.user_id = "user-123"
        mock_org = MagicMock()
        mock_org.id = "org-123"
        mock_org.org_members = [mock_org_member]
        self.mock_org_service.get_org.return_value = mock_org
        self.mock_auth_service.generate_tokens.return_value = Token(
            access_token="access-token",
            refresh_token="refresh-token",
            token_type="bearer",
        )

        response = self.client.post("/auth/login", data={
            "username": "test@example.com",
            "password": "securepassword",
            "org_id": "org-123",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["access_token"], "access-token")

    def test_login_bad_password(self):
        self.mock_user_service.get_user_by_email.return_value = {
            "_id": "user-123",
            "password": "hashed_pass",
        }
        self.mock_user_service.verify_password_async = AsyncMock(return_value=False)

        response = self.client.post("/auth/login", data={
            "username": "test@example.com",
            "password": "wrongpassword",
        })
        self.assertEqual(response.status_code, 401)

    def test_login_missing_org_identification(self):
        self.mock_user_service.get_user_by_email.return_value = {
            "_id": "user-123",
            "password": "hashed_pass",
        }
        self.mock_user_service.verify_password_async = AsyncMock(return_value=True)
        self.mock_org_service.get_user_orgs.return_value = [
            MagicMock(id="org-123", spec=[])
        ]

        response = self.client.post("/auth/login", data={
            "username": "test@example.com",
            "password": "securepassword",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("organization", response.json()["error"]["message"].lower())

    def test_refresh_success(self):
        mock_token_data = MagicMock()
        mock_token_data.user_id = "user-123"
        mock_token_data.org_id = "org-123"
        self.mock_auth_service.verify_token.return_value = mock_token_data
        self.mock_auth_service.generate_tokens.return_value = Token(
            access_token="new-access-token",
            refresh_token="new-refresh-token",
            token_type="bearer",
        )

        response = self.client.post("/auth/refresh", params={"refresh_token": "valid-refresh-token"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["access_token"], "new-access-token")
        self.assertEqual(data["refresh_token"], "new-refresh-token")

    def test_refresh_invalid_token(self):
        self.mock_auth_service.verify_token.return_value = None
        response = self.client.post("/auth/refresh", params={"refresh_token": "invalid-token"})
        self.assertEqual(response.status_code, 401)

    def test_logout_success(self):
        self.mock_auth_service.revoke_token.return_value = None
        response = self.client.post("/auth/logout", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Successfully logged out")
