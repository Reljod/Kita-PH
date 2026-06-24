import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.organization import (
    OrganizationResponse, OrgMember, OrgRole, Integrations
)
from app.models.user import UserResponse


class TestOrganizationRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.organization import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_org_service = MagicMock()

        from app.security import get_current_user, require_org_membership, oauth2_scheme
        from app.dependencies import get_org_service

        cls.current_user = UserResponse(
            id="user-123",
            email="admin@example.com",
            first_name="Admin",
            last_name="User",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        cls.test_app.dependency_overrides[get_current_user] = lambda: cls.current_user
        cls.test_app.dependency_overrides[require_org_membership] = lambda: "org-123"
        cls.test_app.dependency_overrides[get_org_service] = lambda: cls.mock_org_service
        cls.test_app.dependency_overrides[oauth2_scheme] = lambda: "test-token"

        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_create_organization_success(self):
        self.mock_org_service.create_org.return_value = OrganizationResponse(
            id="org-123",
            org_name="Test Org",
            org_code="test-org",
            org_members=[OrgMember(user_id="user-123", role=OrgRole.ADMIN)],
            integrations=Integrations(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.post("/org/", json={
            "org_name": "Test Org",
            "org_code": "test-org",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["org_name"], "Test Org")
        self.assertEqual(data["org_code"], "test-org")

    def test_create_organization_validation_error(self):
        response = self.client.post("/org/", json={"org_name": "", "org_code": ""})
        self.assertEqual(response.status_code, 422)

    def test_get_org_creation_status_success(self):
        self.mock_org_service.get_org.return_value = OrganizationResponse(
            id="org-123",
            org_name="Test Org",
            org_code="test-org",
            status="completed",
            org_members=[OrgMember(user_id="user-123", role=OrgRole.ADMIN)],
            integrations=Integrations(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.get("/org/org-123/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["org_id"], "org-123")
        self.assertEqual(data["status"], "completed")

    def test_get_org_creation_status_not_found(self):
        self.mock_org_service.get_org.return_value = None
        self.mock_org_service.get_org_by_code.return_value = None
        response = self.client.get("/org/org-999/status")
        self.assertEqual(response.status_code, 404)

    def test_get_org_creation_status_forbidden(self):
        self.mock_org_service.get_org.return_value = OrganizationResponse(
            id="org-999",
            org_name="Other Org",
            org_code="other",
            org_members=[OrgMember(user_id="other-user", role=OrgRole.ADMIN)],
            integrations=Integrations(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.get("/org/org-999/status")
        self.assertEqual(response.status_code, 403)

    def test_get_my_organizations_success(self):
        self.mock_org_service.get_user_orgs.return_value = [
            OrganizationResponse(
                id="org-123",
                org_name="Test Org",
                org_code="test-org",
                org_members=[OrgMember(user_id="user-123", role=OrgRole.ADMIN)],
                integrations=Integrations(),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]
        response = self.client.get("/org/me")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["org_name"], "Test Org")

    def test_get_organization_success(self):
        self.mock_org_service.get_org.return_value = OrganizationResponse(
            id="org-123",
            org_name="Test Org",
            org_code="test-org",
            org_members=[OrgMember(user_id="user-123", role=OrgRole.ADMIN)],
            integrations=Integrations(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.get("/org/org-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["org_name"], "Test Org")

    def test_get_organization_not_found(self):
        self.mock_org_service.get_org.return_value = None
        response = self.client.get("/org/org-123")
        self.assertEqual(response.status_code, 404)

    def test_update_organization_success(self):
        self.mock_org_service.update_org.return_value = OrganizationResponse(
            id="org-123",
            org_name="Updated Org",
            org_code="updated-org",
            org_members=[OrgMember(user_id="user-123", role=OrgRole.ADMIN)],
            integrations=Integrations(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.patch("/org/org-123", json={"org_name": "Updated Org"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["org_name"], "Updated Org")

    def test_update_organization_not_found(self):
        self.mock_org_service.update_org.return_value = None
        response = self.client.patch("/org/org-123", json={"org_name": "Updated"})
        self.assertEqual(response.status_code, 404)

    def test_update_integrations_success(self):
        self.mock_org_service.update_integrations.return_value = OrganizationResponse(
            id="org-123",
            org_name="Test Org",
            org_code="test-org",
            org_members=[OrgMember(user_id="user-123", role=OrgRole.ADMIN)],
            integrations=Integrations(facebook_page_id="fb-page-123"),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.patch("/org/org-123/integrations", json={"facebook_page_id": "fb-page-123"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["integrations"]["facebook_page_id"], "fb-page-123")

    def test_update_integrations_not_found(self):
        self.mock_org_service.update_integrations.return_value = None
        response = self.client.patch("/org/org-123/integrations", json={"facebook_page_id": "fb-page-123"})
        self.assertEqual(response.status_code, 404)

    def test_add_or_update_member_success(self):
        self.mock_org_service.add_or_update_member.return_value = OrganizationResponse(
            id="org-123",
            org_name="Test Org",
            org_code="test-org",
            org_members=[
                OrgMember(user_id="user-123", role=OrgRole.ADMIN),
                OrgMember(user_id="user-456", role=OrgRole.DEV),
            ],
            integrations=Integrations(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.put("/org/org-123/member", json={
            "user_id": "user-456",
            "role": "DEV",
        })
        self.assertEqual(response.status_code, 200)
        members = response.json()["org_members"]
        self.assertEqual(len(members), 2)

    def test_add_or_update_member_not_found(self):
        self.mock_org_service.add_or_update_member.return_value = None
        response = self.client.put("/org/org-123/member", json={
            "user_id": "user-456",
            "role": "DEV",
        })
        self.assertEqual(response.status_code, 404)

    def test_revoke_member_success(self):
        self.mock_org_service.revoke_member.return_value = OrganizationResponse(
            id="org-123",
            org_name="Test Org",
            org_code="test-org",
            org_members=[OrgMember(user_id="user-123", role=OrgRole.ADMIN)],
            integrations=Integrations(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        response = self.client.delete("/org/org-123/member/user-456")
        self.assertEqual(response.status_code, 200)
        members = response.json()["org_members"]
        self.assertEqual(len(members), 1)

    def test_revoke_member_not_found(self):
        self.mock_org_service.revoke_member.return_value = None
        response = self.client.delete("/org/org-123/member/user-456")
        self.assertEqual(response.status_code, 404)
