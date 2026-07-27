"""Tests for app.services.auth_service — JWT issuance, verification, revocation.

Token verification is the gate on every authenticated route, so the edge cases
here (expired, tampered, revoked, wrong-secret, missing-claim) matter more than
the happy path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from freezegun import freeze_time
from jose import jwt

from app.models.auth import TokenData
from app.services.auth_service import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    REFRESH_TOKEN_EXPIRE_DAYS,
    SECRET_KEY,
    AuthService,
)

USER = "user_abc123"
ORG = "org_xyz789"


@pytest.fixture
def service() -> AuthService:
    return AuthService()


@pytest.fixture
def tokens_collection():
    """Patch the tokens collection and hand the double back to the test."""
    collection = MagicMock(name="tokens")
    collection.find_one.return_value = {"is_revoked": False}
    with patch("app.services.auth_service.db") as fake_db:
        fake_db.get_tokens_collection.return_value = collection
        yield collection


def decode(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# --- Access tokens --------------------------------------------------------


class TestCreateAccessToken:
    def test_encodes_the_payload(self, service):
        token, _ = service.create_access_token({"sub": USER})
        assert decode(token)["sub"] == USER

    def test_returns_the_expiry_alongside_the_token(self, service):
        token, expires_at = service.create_access_token({"sub": USER})
        assert decode(token)["exp"] == int(expires_at.timestamp())

    @freeze_time("2026-01-15 12:00:00")
    def test_default_expiry_is_the_configured_window(self, service):
        _, expires_at = service.create_access_token({"sub": USER})
        expected = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        assert abs((expires_at - expected).total_seconds()) < 1

    @freeze_time("2026-01-15 12:00:00")
    def test_explicit_expiry_delta_is_honoured(self, service):
        _, expires_at = service.create_access_token(
            {"sub": USER}, expires_delta=timedelta(minutes=5)
        )
        expected = datetime.now(UTC) + timedelta(minutes=5)
        assert abs((expires_at - expected).total_seconds()) < 1

    def test_extra_claims_survive_the_round_trip(self, service):
        token, _ = service.create_access_token({"sub": USER, "org_id": ORG})
        assert decode(token)["org_id"] == ORG

    def test_the_callers_payload_is_not_mutated(self, service):
        payload = {"sub": USER}
        service.create_access_token(payload)
        assert payload == {"sub": USER}, "exp leaked into the caller's dict"

    def test_access_token_is_not_marked_as_a_refresh_token(self, service):
        token, _ = service.create_access_token({"sub": USER})
        assert "type" not in decode(token)


class TestCreateRefreshToken:
    def test_is_tagged_as_a_refresh_token(self, service):
        assert decode(service.create_refresh_token({"sub": USER}))["type"] == "refresh"

    @freeze_time("2026-01-15 12:00:00")
    def test_expires_after_the_configured_number_of_days(self, service):
        exp = decode(service.create_refresh_token({"sub": USER}))["exp"]
        expected = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        assert abs(exp - expected.timestamp()) < 1

    def test_outlives_an_access_token(self, service):
        access, _ = service.create_access_token({"sub": USER})
        refresh = service.create_refresh_token({"sub": USER})
        assert decode(refresh)["exp"] > decode(access)["exp"]

    def test_the_callers_payload_is_not_mutated(self, service):
        payload = {"sub": USER}
        service.create_refresh_token(payload)
        assert payload == {"sub": USER}


# --- Verification ---------------------------------------------------------


class TestVerifyToken:
    def test_accepts_a_valid_unrevoked_token(self, service, tokens_collection):
        token, _ = service.create_access_token({"sub": USER, "org_id": ORG})
        result = service.verify_token(token)
        assert result == TokenData(user_id=USER, org_id=ORG)

    def test_org_id_is_none_when_the_claim_is_absent(self, service, tokens_collection):
        token, _ = service.create_access_token({"sub": USER})
        assert service.verify_token(token).org_id is None

    def test_rejects_a_token_missing_the_sub_claim(self, service, tokens_collection):
        token, _ = service.create_access_token({"org_id": ORG})
        assert service.verify_token(token) is None

    def test_rejects_a_revoked_token(self, service, tokens_collection):
        tokens_collection.find_one.return_value = None
        token, _ = service.create_access_token({"sub": USER})
        assert service.verify_token(token) is None

    def test_rejects_a_token_signed_with_a_different_secret(
        self, service, tokens_collection
    ):
        forged = jwt.encode(
            {"sub": USER, "exp": datetime.now(UTC) + timedelta(hours=1)},
            "an-attackers-secret",
            algorithm=ALGORITHM,
        )
        assert service.verify_token(forged) is None

    def test_rejects_an_expired_token(self, service, tokens_collection):
        expired = jwt.encode(
            {"sub": USER, "exp": datetime.now(UTC) - timedelta(seconds=1)},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        assert service.verify_token(expired) is None

    @pytest.mark.parametrize(
        "garbage", ["", "not.a.jwt", "a.b", "....", "Bearer sometoken"]
    )
    def test_rejects_malformed_tokens(self, service, tokens_collection, garbage):
        assert service.verify_token(garbage) is None

    def test_rejects_an_unsigned_none_algorithm_token(self, service, tokens_collection):
        """The classic JWT downgrade attack must not authenticate anyone."""
        unsigned = (
            jwt.encode({"sub": USER}, "", algorithm="HS256").rsplit(".", 1)[0] + "."
        )
        assert service.verify_token(unsigned) is None

    def test_the_revocation_lookup_checks_both_token_columns(
        self, service, tokens_collection
    ):
        token, _ = service.create_access_token({"sub": USER})
        service.verify_token(token)
        query = tokens_collection.find_one.call_args[0][0]
        assert query["is_revoked"] is False
        assert {"access_token": token} in query["$or"]
        assert {"refresh_token": token} in query["$or"]


# --- Token pair issuance --------------------------------------------------


class TestGenerateTokens:
    def test_returns_a_bearer_token_pair(self, service, tokens_collection):
        result = service.generate_tokens(USER)
        assert result.token_type == "bearer"
        assert result.access_token and result.refresh_token

    def test_both_tokens_carry_the_user_id(self, service, tokens_collection):
        result = service.generate_tokens(USER)
        assert decode(result.access_token)["sub"] == USER
        assert decode(result.refresh_token)["sub"] == USER

    def test_org_id_is_embedded_in_both_tokens_when_supplied(
        self, service, tokens_collection
    ):
        result = service.generate_tokens(USER, ORG)
        assert decode(result.access_token)["org_id"] == ORG
        assert decode(result.refresh_token)["org_id"] == ORG

    def test_org_id_is_omitted_when_not_supplied(self, service, tokens_collection):
        result = service.generate_tokens(USER)
        assert "org_id" not in decode(result.access_token)

    def test_the_pair_is_persisted(self, service, tokens_collection):
        result = service.generate_tokens(USER, ORG)
        stored = tokens_collection.insert_one.call_args[0][0]
        assert stored["user_id"] == USER
        assert stored["access_token"] == result.access_token
        assert stored["refresh_token"] == result.refresh_token

    def test_the_stored_pair_starts_unrevoked(self, service, tokens_collection):
        service.generate_tokens(USER)
        assert tokens_collection.insert_one.call_args[0][0]["is_revoked"] is False

    def test_the_stored_document_records_the_access_token_expiry(
        self, service, tokens_collection
    ):
        service.generate_tokens(USER)
        stored = tokens_collection.insert_one.call_args[0][0]
        assert isinstance(stored["expires_at"], datetime)

    def test_issuing_tokens_emits_no_pydantic_deprecation_warning(
        self, service, tokens_collection, recwarn
    ):
        """Serialising the token document with the v1 `.dict()` shim still
        works but is deprecated; it must not be on the login hot path."""
        service.generate_tokens(USER, ORG)
        deprecations = [
            w
            for w in recwarn
            if issubclass(w.category, DeprecationWarning)
            and "dict" in str(w.message).lower()
        ]
        assert not deprecations, f"deprecated serialisation in use: {deprecations}"


# --- Revocation -----------------------------------------------------------


class TestRevocation:
    def test_revoke_token_matches_either_column(self, service, tokens_collection):
        service.revoke_token("tok")
        query, update = tokens_collection.update_many.call_args[0]
        assert {"access_token": "tok"} in query["$or"]
        assert {"refresh_token": "tok"} in query["$or"]
        assert update == {"$set": {"is_revoked": True}}

    def test_revoke_user_tokens_targets_every_token_for_that_user(
        self, service, tokens_collection
    ):
        service.revoke_user_tokens(USER)
        query, update = tokens_collection.update_many.call_args[0]
        assert query == {"user_id": USER}
        assert update == {"$set": {"is_revoked": True}}

    def test_revoke_token_pair_requires_both_tokens_to_match(
        self, service, tokens_collection
    ):
        service.revoke_token_pair("acc", "ref")
        query, update = tokens_collection.update_one.call_args[0]
        assert query == {"access_token": "acc", "refresh_token": "ref"}
        assert update == {"$set": {"is_revoked": True}}

    def test_revoking_a_token_that_does_not_exist_is_not_an_error(
        self, service, tokens_collection
    ):
        tokens_collection.update_many.return_value = MagicMock(modified_count=0)
        service.revoke_token("never-issued")  # must not raise
