"""Tests for app.routes.organization.

Most handlers here take an `{id}` from the path *and* an `authorized_org_id`
from the caller's token. The gap between those two is the whole security
story: passing another organization's id must never reach that organization's
data, whichever of the two the handler happens to use downstream.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.models.organization import OrganizationResponse, OrgMember, OrgRole
from app.models.user import UserResponse
from app.routes import organization as org_routes
from app.security import get_current_user, require_org_membership
from app.dependencies import get_org_service

from .conftest import override

USER_ID = "user_1"
ORG_ID = "org_abc"
OTHER_ORG = "org_someone_else"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def a_user() -> UserResponse:
    return UserResponse(
        id=USER_ID,
        email="person@example.com",
        first_name="Ada",
        last_name="Lovelace",
        created_at=NOW,
        updated_at=NOW,
    )


def an_org(org_id: str = ORG_ID, members: list[str] | None = None, **overrides):
    payload = {
        "id": org_id,
        "org_name": "Acme",
        "org_code": "acme",
        "org_members": [
            OrgMember(user_id=m, role=OrgRole.MEMBER) for m in (members or [USER_ID])
        ],
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return OrganizationResponse(**payload)


@pytest.fixture
def client(make_client, org_service):
    return make_client(
        org_routes.router,
        {
            get_current_user: override(a_user()),
            get_org_service: override(org_service),
            require_org_membership: override(ORG_ID),
        },
    )


# --- create ---------------------------------------------------------------


class TestCreateOrganization:
    @pytest.fixture(autouse=True)
    def _no_scaffolding(self):
        """Creation queues a background task that builds agents, tools and a
        RAG store. Stub it out — the route's job is to queue it, not to run
        it."""
        with patch.object(org_routes, "run_org_scaffolding", new=MagicMock()):
            yield

    def test_an_organization_is_created(self, client, org_service):
        org_service.create_org.return_value = an_org()
        response = client.post("/org/", json={"org_name": "Acme", "org_code": "acme"})
        assert response.status_code == 200

    def test_the_caller_becomes_the_creator(self, client, org_service):
        org_service.create_org.return_value = an_org()
        client.post("/org/", json={"org_name": "Acme", "org_code": "acme"})
        assert org_service.create_org.call_args[0][1] == USER_ID

    def test_the_submitted_name_and_code_are_used(self, client, org_service):
        """Regression: the handler previously read org_in.name, which does
        not exist on OrgCreate, so every create raised AttributeError and
        returned 500 before doing any work."""
        org_service.create_org.return_value = an_org()
        response = client.post("/org/", json={"org_name": "Acme", "org_code": "acme"})
        assert response.status_code == 200
        submitted = org_service.create_org.call_args[0][0]
        assert submitted.org_name == "Acme" and submitted.org_code == "acme"

    @pytest.mark.parametrize(
        "body",
        [
            {"org_name": "", "org_code": "acme"},
            {"org_name": "Acme", "org_code": ""},
            {"org_name": "Acme"},
            {"org_code": "acme"},
            {"org_name": "x" * 101, "org_code": "acme"},
        ],
    )
    def test_invalid_input_is_rejected(self, client, org_service, body):
        assert client.post("/org/", json=body).status_code == 422
        org_service.create_org.assert_not_called()


# --- status ---------------------------------------------------------------


class TestOrgCreationStatus:
    def test_the_status_is_returned(self, client, org_service):
        org_service.get_org.return_value = an_org(status="initializing")
        response = client.get(f"/org/{ORG_ID}/status")
        assert response.status_code == 200
        assert response.json()["status"] == "initializing"

    def test_status_defaults_to_completed_when_unset(self, client, org_service):
        org_service.get_org.return_value = an_org(status=None)
        assert client.get(f"/org/{ORG_ID}/status").json()["status"] == "completed"

    def test_an_org_code_also_resolves(self, client, org_service):
        org_service.get_org.return_value = None
        org_service.get_org_by_code.return_value = an_org()
        assert client.get("/org/acme/status").status_code == 200

    def test_an_unknown_org_is_a_404(self, client, org_service):
        org_service.get_org.return_value = None
        org_service.get_org_by_code.return_value = None
        assert client.get(f"/org/{ORG_ID}/status").status_code == 404

    def test_a_non_member_is_forbidden(self, client, org_service):
        """Status is a side channel: it would otherwise confirm an
        organization exists to anyone who guesses its code."""
        org_service.get_org.return_value = an_org(members=["somebody_else"])
        assert client.get(f"/org/{ORG_ID}/status").status_code == 403


# --- read -----------------------------------------------------------------


class TestGetMyOrganizations:
    def test_the_callers_organizations_are_listed(self, client, org_service):
        org_service.get_user_orgs.return_value = [an_org(), an_org(org_id="org_2")]
        response = client.get("/org/me")
        assert response.status_code == 200 and len(response.json()) == 2

    def test_the_lookup_is_scoped_to_the_caller(self, client, org_service):
        org_service.get_user_orgs.return_value = []
        client.get("/org/me")
        org_service.get_user_orgs.assert_called_once_with(USER_ID)

    def test_a_user_with_no_organizations_gets_an_empty_list(self, client, org_service):
        org_service.get_user_orgs.return_value = []
        assert client.get("/org/me").json() == []


class TestGetOrganization:
    def test_the_authorized_organization_is_returned(self, client, org_service):
        org_service.get_org.return_value = an_org()
        assert client.get(f"/org/{ORG_ID}").status_code == 200

    def test_another_organizations_id_is_forbidden(self, client, org_service):
        """The path id and the token's org must agree."""
        org_service.get_org_by_code.return_value = None
        assert client.get(f"/org/{OTHER_ORG}").status_code == 403

    def test_a_code_belonging_to_the_authorized_org_is_accepted(
        self, client, org_service
    ):
        org_service.get_org_by_code.return_value = an_org()
        org_service.get_org.return_value = an_org()
        assert client.get("/org/acme").status_code == 200

    def test_a_code_belonging_to_another_org_is_forbidden(self, client, org_service):
        org_service.get_org_by_code.return_value = an_org(org_id=OTHER_ORG)
        assert client.get("/org/acme").status_code == 403

    def test_the_data_returned_is_the_token_org_not_the_path_id(
        self, client, org_service
    ):
        """Even on the happy path the handler must read the authorized org,
        so a mismatched path id can never select the data."""
        org_service.get_org_by_code.return_value = an_org()
        org_service.get_org.return_value = an_org()
        client.get("/org/acme")
        assert org_service.get_org.call_args[0][0] == ORG_ID

    def test_a_vanished_organization_is_a_404(self, client, org_service):
        org_service.get_org.return_value = None
        assert client.get(f"/org/{ORG_ID}").status_code == 404


# --- update ---------------------------------------------------------------


class TestUpdateOrganization:
    def test_the_organization_is_updated(self, client, org_service):
        org_service.update_org.return_value = an_org(org_name="Renamed")
        response = client.patch(f"/org/{ORG_ID}", json={"org_name": "Renamed"})
        assert response.status_code == 200

    def test_the_update_targets_the_authorized_org(self, client, org_service):
        org_service.update_org.return_value = an_org()
        client.patch(f"/org/{ORG_ID}", json={"org_name": "Renamed"})
        assert org_service.update_org.call_args[0][0] == ORG_ID

    def test_updating_another_organization_is_forbidden(self, client, org_service):
        org_service.get_org_by_code.return_value = None
        response = client.patch(f"/org/{OTHER_ORG}", json={"org_name": "Renamed"})
        assert response.status_code == 403

    def test_a_forbidden_update_writes_nothing(self, client, org_service):
        org_service.get_org_by_code.return_value = None
        client.patch(f"/org/{OTHER_ORG}", json={"org_name": "Renamed"})
        org_service.update_org.assert_not_called()

    def test_a_vanished_organization_is_a_404(self, client, org_service):
        org_service.update_org.return_value = None
        response = client.patch(f"/org/{ORG_ID}", json={"org_name": "Renamed"})
        assert response.status_code == 404

    def test_an_empty_name_is_rejected(self, client, org_service):
        assert client.patch(f"/org/{ORG_ID}", json={"org_name": ""}).status_code == 422


class TestUpdateIntegrations:
    def test_integrations_are_updated(self, client, org_service):
        org_service.update_integrations.return_value = an_org()
        response = client.patch(
            f"/org/{ORG_ID}/integrations", json={"facebook_page_id": "42"}
        )
        assert response.status_code == 200

    def test_the_update_targets_the_authorized_org(self, client, org_service):
        org_service.update_integrations.return_value = an_org()
        client.patch(f"/org/{ORG_ID}/integrations", json={"facebook_page_id": "42"})
        assert org_service.update_integrations.call_args[0][0] == ORG_ID

    def test_updating_another_organization_is_forbidden(self, client, org_service):
        org_service.get_org_by_code.return_value = None
        response = client.patch(
            f"/org/{OTHER_ORG}/integrations", json={"facebook_page_id": "42"}
        )
        assert response.status_code == 403

    def test_a_vanished_organization_is_a_404(self, client, org_service):
        org_service.update_integrations.return_value = None
        response = client.patch(
            f"/org/{ORG_ID}/integrations", json={"facebook_page_id": "42"}
        )
        assert response.status_code == 404


# --- membership -----------------------------------------------------------


class TestAddOrUpdateMember:
    def test_a_member_is_added(self, client, org_service):
        org_service.add_or_update_member.return_value = an_org()
        response = client.put(
            f"/org/{ORG_ID}/member", json={"user_id": "u2", "role": "MEMBER"}
        )
        assert response.status_code == 200

    def test_the_change_targets_the_authorized_org(self, client, org_service):
        org_service.add_or_update_member.return_value = an_org()
        client.put(f"/org/{ORG_ID}/member", json={"user_id": "u2", "role": "MEMBER"})
        assert org_service.add_or_update_member.call_args[0][0] == ORG_ID

    def test_adding_to_another_organization_is_forbidden(self, client, org_service):
        """Otherwise anyone could add themselves to any organization."""
        org_service.get_org_by_code.return_value = None
        response = client.put(
            f"/org/{OTHER_ORG}/member", json={"user_id": "u2", "role": "MEMBER"}
        )
        assert response.status_code == 403

    def test_a_forbidden_add_writes_nothing(self, client, org_service):
        org_service.get_org_by_code.return_value = None
        client.put(f"/org/{OTHER_ORG}/member", json={"user_id": "u2", "role": "MEMBER"})
        org_service.add_or_update_member.assert_not_called()

    def test_an_unknown_role_is_rejected(self, client, org_service):
        response = client.put(
            f"/org/{ORG_ID}/member", json={"user_id": "u2", "role": "SUPERUSER"}
        )
        assert response.status_code == 422

    def test_a_missing_user_id_is_rejected(self, client):
        assert (
            client.put(f"/org/{ORG_ID}/member", json={"role": "MEMBER"}).status_code
            == 422
        )

    def test_a_vanished_organization_is_a_404(self, client, org_service):
        org_service.add_or_update_member.return_value = None
        response = client.put(
            f"/org/{ORG_ID}/member", json={"user_id": "u2", "role": "MEMBER"}
        )
        assert response.status_code == 404


class TestRevokeMember:
    def test_a_member_is_revoked(self, client, org_service):
        org_service.revoke_member.return_value = an_org(members=[])
        assert client.delete(f"/org/{ORG_ID}/member/u2").status_code == 200

    def test_the_revocation_targets_the_authorized_org(self, client, org_service):
        org_service.revoke_member.return_value = an_org()
        client.delete(f"/org/{ORG_ID}/member/u2")
        assert org_service.revoke_member.call_args[0][0] == ORG_ID

    def test_revoking_from_another_organization_is_forbidden(self, client, org_service):
        org_service.get_org_by_code.return_value = None
        assert client.delete(f"/org/{OTHER_ORG}/member/u2").status_code == 403

    def test_a_forbidden_revoke_writes_nothing(self, client, org_service):
        org_service.get_org_by_code.return_value = None
        client.delete(f"/org/{OTHER_ORG}/member/u2")
        org_service.revoke_member.assert_not_called()

    def test_a_vanished_organization_is_a_404(self, client, org_service):
        org_service.revoke_member.return_value = None
        assert client.delete(f"/org/{ORG_ID}/member/u2").status_code == 404
