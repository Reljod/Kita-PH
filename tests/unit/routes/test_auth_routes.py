"""Tests for app.routes.auth — register, login, refresh, logout.

Login is the densest branch in the codebase: it resolves an organization from
either an id or a code, checks membership, and has a special case for users
who belong to no organization at all. Every rejection has to look identical
from outside, or the endpoint becomes an oracle for which emails and org codes
exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId

from app.models.auth import TokenData
from app.models.organization import OrganizationResponse, OrgMember, OrgRole
from app.models.user import UserResponse
from app.routes import auth as auth_routes
from app.security import get_auth_service, get_org_service, get_user_service

from .conftest import override

OID = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")
USER_ID = str(OID)
ORG_ID = "org_abc"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)
AMBIGUOUS = "Incorrect email, password, or organization code"


def error_message(response) -> str:
    """Pull the message out of the project's error envelope.

    setup_error_handlers rewrites every HTTPException into
    {"error": {"code", "message", "details", "trace_id"}}, so `detail` is
    not what a client actually receives.
    """
    return response.json()["error"]["message"]


def user_record(**overrides) -> dict:
    record = {"_id": OID, "email": "person@example.com", "password": "$2b$12$hash"}
    record.update(overrides)
    return record


def user_response() -> UserResponse:
    return UserResponse(
        id=USER_ID,
        email="person@example.com",
        first_name="Ada",
        last_name="Lovelace",
        created_at=NOW,
        updated_at=NOW,
    )


def an_org(members: list[str], org_id: str = ORG_ID) -> OrganizationResponse:
    return OrganizationResponse(
        id=org_id,
        org_name="Acme",
        org_code="acme",
        org_members=[OrgMember(user_id=m, role=OrgRole.MEMBER) for m in members],
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def client(make_client, auth_service, user_service, org_service):
    return make_client(
        auth_routes.router,
        {
            get_auth_service: override(auth_service),
            get_user_service: override(user_service),
            get_org_service: override(org_service),
        },
    )


# --- register -------------------------------------------------------------


class TestRegister:
    @pytest.fixture(autouse=True)
    def _defaults(self, user_service):
        user_service.hash_password_async = AsyncMock(return_value="$2b$12$hashed")
        user_service.create_user.return_value = user_response()

    def payload(self, **overrides) -> dict:
        body = {
            "email": "person@example.com",
            "password": "hunter2000",
            "first_name": "Ada",
            "last_name": "Lovelace",
        }
        body.update(overrides)
        return body

    def test_a_new_email_is_registered(self, client):
        assert client.post("/auth/register", json=self.payload()).status_code == 200

    def test_a_token_pair_is_returned(self, client):
        body = client.post("/auth/register", json=self.payload()).json()
        assert body["access_token"] and body["refresh_token"]
        assert body["token_type"] == "bearer"

    def test_the_password_is_hashed_off_the_event_loop(self, client, user_service):
        client.post("/auth/register", json=self.payload())
        user_service.hash_password_async.assert_awaited_once_with("hunter2000")

    def test_the_stored_password_is_the_hash_not_the_plaintext(
        self, client, user_service
    ):
        client.post("/auth/register", json=self.payload())
        created = user_service.create_user.call_args[0][0]
        assert created.password == "$2b$12$hashed"

    def test_the_user_is_created_as_prehashed(self, client, user_service):
        """Re-hashing an already-hashed password would produce a digest that
        can never be authenticated against."""
        client.post("/auth/register", json=self.payload())
        assert user_service.create_user.call_args.kwargs["prehashed"] is True

    def test_a_duplicate_email_is_rejected(self, client, user_service):
        user_service.get_user_by_email.return_value = user_record()
        assert client.post("/auth/register", json=self.payload()).status_code == 400

    def test_a_duplicate_email_does_not_confirm_the_address_exists(
        self, client, user_service
    ):
        """Otherwise registration enumerates who already has an account."""
        user_service.get_user_by_email.return_value = user_record()
        message = error_message(client.post("/auth/register", json=self.payload()))
        assert "already" not in message.lower()
        assert "exist" not in message.lower()

    def test_a_duplicate_email_creates_nothing(self, client, user_service):
        user_service.get_user_by_email.return_value = user_record()
        client.post("/auth/register", json=self.payload())
        user_service.create_user.assert_not_called()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("email", "not-an-email"),
            ("password", "short"),
            ("first_name", ""),
            ("last_name", ""),
        ],
    )
    def test_invalid_input_is_rejected_before_any_work(
        self, client, user_service, field, value
    ):
        response = client.post("/auth/register", json=self.payload(**{field: value}))
        assert response.status_code == 422
        user_service.create_user.assert_not_called()

    def test_a_missing_field_is_rejected(self, client):
        body = self.payload()
        del body["email"]
        assert client.post("/auth/register", json=body).status_code == 422


# --- login ----------------------------------------------------------------


class TestLogin:
    @pytest.fixture(autouse=True)
    def _password_ok(self, user_service):
        user_service.verify_password_async = AsyncMock(return_value=True)

    def form(self, **overrides) -> dict:
        data = {"username": "person@example.com", "password": "hunter2000"}
        data.update(overrides)
        return data

    def test_a_user_with_no_organizations_may_log_in(self, client, user_service):
        """Registration issues an org-less account; it has to be able to log
        in before it can create or join an organization."""
        user_service.get_user_by_email.return_value = user_record()
        assert client.post("/auth/login", data=self.form()).status_code == 200

    def test_an_org_less_login_issues_a_token_without_an_org(
        self, client, user_service, auth_service
    ):
        user_service.get_user_by_email.return_value = user_record()
        client.post("/auth/login", data=self.form())
        auth_service.generate_tokens.assert_called_once_with(USER_ID, None)

    def test_login_by_org_code_succeeds(self, client, user_service, org_service):
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_user_orgs.return_value = [an_org([USER_ID])]
        org_service.get_org_by_code.return_value = an_org([USER_ID])
        org_service.get_org.return_value = an_org([USER_ID])
        response = client.post("/auth/login", data=self.form(org_code="acme"))
        assert response.status_code == 200

    def test_the_token_is_scoped_to_the_resolved_org(
        self, client, user_service, org_service, auth_service
    ):
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_user_orgs.return_value = [an_org([USER_ID])]
        org_service.get_org_by_code.return_value = an_org([USER_ID])
        org_service.get_org.return_value = an_org([USER_ID])
        client.post("/auth/login", data=self.form(org_code="acme"))
        auth_service.generate_tokens.assert_called_once_with(USER_ID, ORG_ID)

    def test_login_by_org_id_succeeds(self, client, user_service, org_service):
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_user_orgs.return_value = [an_org([USER_ID])]
        org_service.get_org.return_value = an_org([USER_ID])
        response = client.post("/auth/login", data=self.form(org_id=ORG_ID))
        assert response.status_code == 200

    def test_an_org_code_takes_precedence_over_an_org_id(
        self, client, user_service, org_service
    ):
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_user_orgs.return_value = [an_org([USER_ID])]
        org_service.get_org_by_code.return_value = an_org([USER_ID], org_id="from-code")
        org_service.get_org.return_value = an_org([USER_ID], org_id="from-code")
        client.post("/auth/login", data=self.form(org_code="acme", org_id="from-id"))
        org_service.get_org.assert_called_once_with("from-code")

    def test_an_unknown_email_is_rejected(self, client, user_service):
        user_service.get_user_by_email.return_value = None
        assert client.post("/auth/login", data=self.form()).status_code == 401

    def test_a_wrong_password_is_rejected(self, client, user_service):
        user_service.get_user_by_email.return_value = user_record()
        user_service.verify_password_async = AsyncMock(return_value=False)
        assert client.post("/auth/login", data=self.form()).status_code == 401

    def test_an_unknown_org_code_is_rejected(self, client, user_service, org_service):
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_org_by_code.return_value = None
        response = client.post("/auth/login", data=self.form(org_code="nope"))
        assert response.status_code == 401

    def test_an_unknown_org_id_is_rejected(self, client, user_service, org_service):
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_org.return_value = None
        response = client.post("/auth/login", data=self.form(org_id="nope"))
        assert response.status_code == 401

    def test_a_non_member_is_rejected(self, client, user_service, org_service):
        """Naming a real organization you don't belong to must not log you
        into it."""
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_org.return_value = an_org(["somebody_else"])
        response = client.post("/auth/login", data=self.form(org_id=ORG_ID))
        assert response.status_code == 401

    def test_a_non_member_gets_no_token(
        self, client, user_service, org_service, auth_service
    ):
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_org.return_value = an_org(["somebody_else"])
        client.post("/auth/login", data=self.form(org_id=ORG_ID))
        auth_service.generate_tokens.assert_not_called()

    @pytest.mark.parametrize(
        "setup",
        ["unknown_email", "wrong_password", "unknown_code", "not_a_member"],
    )
    def test_every_rejection_reads_the_same_from_outside(
        self, client, user_service, org_service, setup
    ):
        """Distinguishable messages would let a caller enumerate accounts and
        organization codes."""
        form = self.form()
        if setup == "unknown_email":
            user_service.get_user_by_email.return_value = None
        elif setup == "wrong_password":
            user_service.get_user_by_email.return_value = user_record()
            user_service.verify_password_async = AsyncMock(return_value=False)
        elif setup == "unknown_code":
            user_service.get_user_by_email.return_value = user_record()
            org_service.get_org_by_code.return_value = None
            form = self.form(org_code="nope")
        else:
            user_service.get_user_by_email.return_value = user_record()
            org_service.get_org.return_value = an_org(["somebody_else"])
            form = self.form(org_id=ORG_ID)

        response = client.post("/auth/login", data=form)
        assert response.status_code == 401
        assert error_message(response) == AMBIGUOUS

    def test_a_member_of_orgs_must_say_which_one(
        self, client, user_service, org_service
    ):
        """Silently picking one would log the user into an arbitrary tenant."""
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_user_orgs.return_value = [an_org([USER_ID])]
        response = client.post("/auth/login", data=self.form())
        assert response.status_code == 400

    def test_the_missing_org_error_says_what_is_missing(
        self, client, user_service, org_service
    ):
        """This one is safe to be specific about: the caller has already
        authenticated, so it leaks nothing."""
        user_service.get_user_by_email.return_value = user_record()
        org_service.get_user_orgs.return_value = [an_org([USER_ID])]
        message = error_message(client.post("/auth/login", data=self.form()))
        assert "org_id" in message or "organization" in message.lower()

    def test_a_401_advertises_bearer_auth(self, client, user_service):
        user_service.get_user_by_email.return_value = None
        response = client.post("/auth/login", data=self.form())
        assert response.headers.get("www-authenticate") == "Bearer"

    def test_a_missing_password_is_rejected_by_validation(self, client):
        response = client.post("/auth/login", data={"username": "a@b.com"})
        assert response.status_code == 422


# --- refresh --------------------------------------------------------------


class TestRefresh:
    def test_a_valid_refresh_token_yields_a_new_pair(self, client, auth_service):
        auth_service.verify_token.return_value = TokenData(
            user_id=USER_ID, org_id=ORG_ID
        )
        response = client.post("/auth/refresh", params={"refresh_token": "tok"})
        assert response.status_code == 200

    def test_the_new_pair_keeps_the_organization_scope(self, client, auth_service):
        auth_service.verify_token.return_value = TokenData(
            user_id=USER_ID, org_id=ORG_ID
        )
        client.post("/auth/refresh", params={"refresh_token": "tok"})
        auth_service.generate_tokens.assert_called_once_with(USER_ID, ORG_ID)

    def test_the_old_token_is_revoked(self, client, auth_service):
        """Leaving it valid would mean a stolen refresh token stays usable
        after the rightful owner has rotated it."""
        auth_service.verify_token.return_value = TokenData(user_id=USER_ID)
        client.post("/auth/refresh", params={"refresh_token": "tok"})
        auth_service.revoke_token.assert_called_once_with("tok")

    def test_an_invalid_refresh_token_is_rejected(self, client, auth_service):
        auth_service.verify_token.return_value = None
        response = client.post("/auth/refresh", params={"refresh_token": "bad"})
        assert response.status_code == 401

    def test_an_invalid_refresh_token_issues_nothing(self, client, auth_service):
        auth_service.verify_token.return_value = None
        client.post("/auth/refresh", params={"refresh_token": "bad"})
        auth_service.generate_tokens.assert_not_called()

    def test_an_invalid_refresh_token_revokes_nothing(self, client, auth_service):
        auth_service.verify_token.return_value = None
        client.post("/auth/refresh", params={"refresh_token": "bad"})
        auth_service.revoke_token.assert_not_called()

    def test_the_refresh_token_is_required(self, client):
        assert client.post("/auth/refresh").status_code == 422


# --- logout ---------------------------------------------------------------


class TestLogout:
    def test_logout_succeeds(self, client):
        response = client.post("/auth/logout", headers={"Authorization": "Bearer tok"})
        assert response.status_code == 200

    def test_the_presented_token_is_revoked(self, client, auth_service):
        client.post("/auth/logout", headers={"Authorization": "Bearer tok"})
        auth_service.revoke_token.assert_called_once_with("tok")

    def test_logout_requires_a_token(self, client, auth_service):
        assert client.post("/auth/logout").status_code == 401
        auth_service.revoke_token.assert_not_called()
