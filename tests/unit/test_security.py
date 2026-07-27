"""Tests for app.security — the dependency chain guarding every route.

These four dependencies decide whether a request is allowed to touch an
organization's data, so the failure modes matter more than the successes:
a missing org claim, a revoked token, a user that no longer exists, and a
caller who is authenticated but not a member.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.exceptions import AuthSessionExpiredError
from app.models.auth import TokenData
from app.models.organization import OrganizationResponse, OrgMember, OrgRole
from app.models.user import UserResponse
from app.security import (
    get_auth_service,
    get_current_org_id,
    get_current_user,
    get_org_service,
    get_token_data,
    get_user_service,
    require_org_membership,
)

USER_ID = "64b7f1c2e4b0a1a2b3c4d5e6"
ORG_ID = "org_abc"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def a_user(user_id: str = USER_ID) -> UserResponse:
    return UserResponse(
        id=user_id,
        email="person@example.com",
        first_name="Ada",
        last_name="Lovelace",
        created_at=NOW,
        updated_at=NOW,
    )


def an_org(members: list[str]) -> OrganizationResponse:
    return OrganizationResponse(
        id=ORG_ID,
        org_name="Acme",
        org_code="acme",
        org_members=[OrgMember(user_id=m, role=OrgRole.MEMBER) for m in members],
        created_at=NOW,
        updated_at=NOW,
    )


class TestServiceProviders:
    """The provider functions exist so routes can be overridden in tests."""

    def test_auth_service_provider_returns_a_service(self):
        assert get_auth_service() is not None

    def test_each_call_yields_an_independent_instance(self):
        assert get_auth_service() is not get_auth_service()

    def test_org_service_provider_returns_a_service(self, patched_db):
        assert get_org_service() is not None

    def test_user_service_provider_returns_a_service(self, patched_db):
        assert get_user_service() is not None


class TestGetTokenData:
    async def test_returns_the_decoded_claims(self):
        auth = MagicMock()
        auth.verify_token.return_value = TokenData(user_id=USER_ID, org_id=ORG_ID)
        result = await get_token_data(token="a.token", auth_service=auth)
        assert result.user_id == USER_ID and result.org_id == ORG_ID

    async def test_the_raw_token_is_what_gets_verified(self):
        auth = MagicMock()
        auth.verify_token.return_value = TokenData(user_id=USER_ID)
        await get_token_data(token="a.token", auth_service=auth)
        auth.verify_token.assert_called_once_with("a.token")

    async def test_a_rejected_token_raises_session_expired(self):
        auth = MagicMock()
        auth.verify_token.return_value = None
        with pytest.raises(AuthSessionExpiredError):
            await get_token_data(token="bad", auth_service=auth)

    async def test_the_session_expired_error_is_a_401(self):
        auth = MagicMock()
        auth.verify_token.return_value = None
        with pytest.raises(AuthSessionExpiredError) as excinfo:
            await get_token_data(token="bad", auth_service=auth)
        assert excinfo.value.status_code == 401

    async def test_a_token_without_an_org_is_still_valid(self):
        """Registration issues an org-less token; it must authenticate so the
        user can go on to create or join an organization."""
        auth = MagicMock()
        auth.verify_token.return_value = TokenData(user_id=USER_ID, org_id=None)
        assert (await get_token_data(token="t", auth_service=auth)).org_id is None


class TestGetCurrentUser:
    async def test_returns_the_user(self):
        users = MagicMock()
        users.get_user_by_id.return_value = a_user()
        result = await get_current_user(
            token_data=TokenData(user_id=USER_ID, org_id=ORG_ID), user_service=users
        )
        assert result.id == USER_ID

    async def test_looks_the_user_up_by_the_token_subject(self):
        users = MagicMock()
        users.get_user_by_id.return_value = a_user()
        await get_current_user(
            token_data=TokenData(user_id=USER_ID), user_service=users
        )
        users.get_user_by_id.assert_called_once_with(USER_ID)

    async def test_a_deleted_user_gets_a_401_not_a_500(self):
        """A still-valid token for a user who has since been removed must read
        as unauthenticated."""
        users = MagicMock()
        users.get_user_by_id.return_value = None
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(
                token_data=TokenData(user_id=USER_ID), user_service=users
            )
        assert excinfo.value.status_code == 401

    async def test_the_401_advertises_bearer_auth(self):
        users = MagicMock()
        users.get_user_by_id.return_value = None
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(
                token_data=TokenData(user_id=USER_ID), user_service=users
            )
        assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"

    async def test_a_malformed_subject_is_unauthenticated_rather_than_a_crash(self):
        """UserService returns None for an unparseable id; the dependency must
        turn that into a 401 rather than letting anything escape."""
        users = MagicMock()
        users.get_user_by_id.return_value = None
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(
                token_data=TokenData(user_id="not-an-object-id"), user_service=users
            )
        assert excinfo.value.status_code == 401


class TestGetCurrentOrgId:
    async def test_returns_the_org_from_the_token(self):
        result = await get_current_org_id(TokenData(user_id=USER_ID, org_id=ORG_ID))
        assert result == ORG_ID

    async def test_a_token_without_an_org_is_forbidden(self):
        with pytest.raises(HTTPException) as excinfo:
            await get_current_org_id(TokenData(user_id=USER_ID, org_id=None))
        assert excinfo.value.status_code == 403

    async def test_an_empty_org_claim_is_also_forbidden(self):
        with pytest.raises(HTTPException) as excinfo:
            await get_current_org_id(TokenData(user_id=USER_ID, org_id=""))
        assert excinfo.value.status_code == 403

    async def test_the_message_tells_the_client_to_pick_an_organization(self):
        with pytest.raises(HTTPException) as excinfo:
            await get_current_org_id(TokenData(user_id=USER_ID, org_id=None))
        assert "organization" in excinfo.value.detail.lower()


class TestRequireOrgMembership:
    async def test_a_member_is_allowed_through(self):
        orgs = MagicMock()
        orgs.get_org.return_value = an_org([USER_ID])
        result = await require_org_membership(
            org_id=ORG_ID, current_user=a_user(), org_service=orgs
        )
        assert result == ORG_ID

    async def test_a_non_member_is_forbidden(self):
        """The core tenant check: holding a token that names an org is not the
        same as belonging to it."""
        orgs = MagicMock()
        orgs.get_org.return_value = an_org(["someone_else"])
        with pytest.raises(HTTPException) as excinfo:
            await require_org_membership(
                org_id=ORG_ID, current_user=a_user(), org_service=orgs
            )
        assert excinfo.value.status_code == 403

    async def test_an_unknown_org_is_forbidden(self):
        orgs = MagicMock()
        orgs.get_org.return_value = None
        with pytest.raises(HTTPException) as excinfo:
            await require_org_membership(
                org_id=ORG_ID, current_user=a_user(), org_service=orgs
            )
        assert excinfo.value.status_code == 403

    async def test_an_org_with_no_members_admits_nobody(self):
        orgs = MagicMock()
        orgs.get_org.return_value = an_org([])
        with pytest.raises(HTTPException):
            await require_org_membership(
                org_id=ORG_ID, current_user=a_user(), org_service=orgs
            )

    async def test_membership_is_checked_against_the_authenticated_user(self):
        orgs = MagicMock()
        orgs.get_org.return_value = an_org(["other_1", USER_ID, "other_2"])
        assert (
            await require_org_membership(
                org_id=ORG_ID, current_user=a_user(), org_service=orgs
            )
            == ORG_ID
        )

    async def test_the_org_looked_up_is_the_one_from_the_token(self):
        orgs = MagicMock()
        orgs.get_org.return_value = an_org([USER_ID])
        await require_org_membership(
            org_id=ORG_ID, current_user=a_user(), org_service=orgs
        )
        orgs.get_org.assert_called_once_with(ORG_ID)
