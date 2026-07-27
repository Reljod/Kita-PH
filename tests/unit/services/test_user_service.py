"""Tests for app.services.user_service — password hashing and user CRUD."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from app.models.user import PasswordUpdate, UserCreate, UserResponse, UserUpdate
from app.services.user_service import UserService

OID = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")


@pytest.fixture
def users_collection() -> MagicMock:
    collection = MagicMock(name="users")
    collection.find_one.return_value = None
    result = MagicMock()
    result.inserted_id = OID
    collection.insert_one.return_value = result
    collection.find_one_and_update.return_value = None
    return collection


@pytest.fixture
def service(users_collection) -> UserService:
    with patch("app.services.user_service.Database") as database:
        database.get_users_collection.return_value = users_collection
        yield UserService()


def user_doc(**overrides) -> dict:
    doc = {
        "_id": OID,
        "email": "person@example.com",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "password": "$2b$12$hashed",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    doc.update(overrides)
    return doc


# --- Password hashing -----------------------------------------------------


class TestPasswordHashing:
    def test_hash_does_not_return_the_plaintext(self, service):
        assert service.hash_password("hunter2000") != "hunter2000"

    def test_hash_is_a_bcrypt_digest(self, service):
        assert service.hash_password("hunter2000").startswith("$2b$")

    def test_hashing_the_same_password_twice_gives_different_digests(self, service):
        """bcrypt salts each hash; identical digests would let an attacker
        spot shared passwords straight from a database dump."""
        assert service.hash_password("same") != service.hash_password("same")

    def test_verify_accepts_the_correct_password(self, service):
        assert service.verify_password(
            "hunter2000", service.hash_password("hunter2000")
        )

    def test_verify_rejects_the_wrong_password(self, service):
        assert not service.verify_password("wrong", service.hash_password("hunter2000"))

    def test_verify_is_case_sensitive(self, service):
        assert not service.verify_password(
            "HUNTER2000", service.hash_password("hunter2000")
        )

    @pytest.mark.parametrize(
        "password", ["", "a", "ünïcödé-påss", "x" * 72, " leading-space"]
    )
    def test_round_trip_for_awkward_passwords(self, service, password):
        assert service.verify_password(password, service.hash_password(password))

    def test_verify_rejects_a_malformed_hash(self, service):
        with pytest.raises(ValueError):
            service.verify_password("anything", "not-a-bcrypt-hash")


class TestAsyncPasswordHashing:
    async def test_async_hash_matches_the_sync_verifier(self, service):
        digest = await service.hash_password_async("hunter2000")
        assert service.verify_password("hunter2000", digest)

    async def test_async_verify_accepts_the_correct_password(self, service):
        digest = service.hash_password("hunter2000")
        assert await service.verify_password_async("hunter2000", digest)

    async def test_async_verify_rejects_the_wrong_password(self, service):
        digest = service.hash_password("hunter2000")
        assert not await service.verify_password_async("wrong", digest)

    async def test_async_hash_does_not_block_the_event_loop(self, service):
        """The whole point of the executor offload: bcrypt must not stall
        concurrent work (this regressed into Cloudflare 502s once already)."""
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(50):
                await asyncio.sleep(0.001)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        await service.hash_password_async("hunter2000")
        await tick_task
        assert ticks > 0, "event loop was blocked for the whole hash"

    async def test_concurrent_hashes_all_complete(self, service):
        digests = await asyncio.gather(
            *[service.hash_password_async(f"pw-{i}") for i in range(4)]
        )
        assert len(set(digests)) == 4


# --- Create ---------------------------------------------------------------


class TestCreateUser:
    @pytest.fixture
    def payload(self) -> UserCreate:
        return UserCreate(
            email="person@example.com",
            password="hunter2000",
            first_name="Ada",
            last_name="Lovelace",
        )

    def test_returns_a_user_response(self, service, payload):
        assert isinstance(service.create_user(payload), UserResponse)

    def test_id_is_the_stringified_inserted_id(self, service, payload):
        assert service.create_user(payload).id == str(OID)

    def test_the_stored_password_is_hashed(self, service, users_collection, payload):
        service.create_user(payload)
        stored = users_collection.insert_one.call_args[0][0]["password"]
        assert stored != "hunter2000"
        assert stored.startswith("$2b$")

    def test_prehashed_password_is_stored_verbatim(
        self, service, users_collection, payload
    ):
        """The async login path hashes upstream; re-hashing would produce a
        digest nobody can ever authenticate against."""
        payload.password = "$2b$12$already-hashed-upstream"
        service.create_user(payload, prehashed=True)
        stored = users_collection.insert_one.call_args[0][0]["password"]
        assert stored == "$2b$12$already-hashed-upstream"

    def test_timestamps_are_set(self, service, users_collection, payload):
        service.create_user(payload)
        stored = users_collection.insert_one.call_args[0][0]
        assert isinstance(stored["created_at"], datetime)
        assert isinstance(stored["updated_at"], datetime)

    def test_the_response_never_carries_the_password(self, service, payload):
        assert not hasattr(service.create_user(payload), "password")


# --- Read -----------------------------------------------------------------


class TestGetUser:
    def test_by_email_returns_the_raw_document(self, service, users_collection):
        users_collection.find_one.return_value = user_doc()
        assert service.get_user_by_email("person@example.com")["id"] == str(OID)

    def test_by_email_queries_on_email(self, service, users_collection):
        service.get_user_by_email("person@example.com")
        users_collection.find_one.assert_called_once_with(
            {"email": "person@example.com"}
        )

    def test_by_email_returns_none_when_absent(self, service, users_collection):
        users_collection.find_one.return_value = None
        assert service.get_user_by_email("nobody@example.com") is None

    def test_by_email_keeps_the_password_hash_for_the_login_path(
        self, service, users_collection
    ):
        """get_user_by_email is what login compares against, so unlike the
        response model it must retain the hash."""
        users_collection.find_one.return_value = user_doc()
        assert "password" in service.get_user_by_email("person@example.com")

    def test_by_id_returns_a_user_response(self, service, users_collection):
        users_collection.find_one.return_value = user_doc()
        result = service.get_user_by_id(str(OID))
        assert isinstance(result, UserResponse) and result.id == str(OID)

    def test_by_id_queries_on_object_id(self, service, users_collection):
        service.get_user_by_id(str(OID))
        users_collection.find_one.assert_called_once_with({"_id": OID})

    def test_by_id_returns_none_when_absent(self, service, users_collection):
        users_collection.find_one.return_value = None
        assert service.get_user_by_id(str(OID)) is None

    @pytest.mark.parametrize(
        "bad_id", ["", "not-an-object-id", "12345", "z" * 24, "  "]
    )
    def test_by_id_returns_none_for_a_malformed_id(
        self, service, users_collection, bad_id
    ):
        """A malformed `sub` claim in a JWT reaches this method. It must read
        as 'no such user' (401), not explode into a 500."""
        assert service.get_user_by_id(bad_id) is None


# --- Update ---------------------------------------------------------------


class TestUpdateUser:
    def test_returns_the_updated_user(self, service, users_collection):
        users_collection.find_one_and_update.return_value = user_doc(first_name="Grace")
        result = service.update_user(str(OID), UserUpdate(first_name="Grace"))
        assert result.first_name == "Grace"

    def test_only_set_fields_are_written(self, service, users_collection):
        users_collection.find_one_and_update.return_value = user_doc()
        service.update_user(str(OID), UserUpdate(first_name="Grace"))
        update = users_collection.find_one_and_update.call_args[0][1]["$set"]
        assert "last_name" not in update, "unset fields must not be nulled out"

    def test_updated_at_is_refreshed(self, service, users_collection):
        users_collection.find_one_and_update.return_value = user_doc()
        service.update_user(str(OID), UserUpdate(first_name="Grace"))
        update = users_collection.find_one_and_update.call_args[0][1]["$set"]
        assert isinstance(update["updated_at"], datetime)

    def test_returns_none_when_the_user_is_absent(self, service, users_collection):
        users_collection.find_one_and_update.return_value = None
        assert service.update_user(str(OID), UserUpdate(first_name="Grace")) is None

    def test_returns_none_for_a_malformed_id(self, service, users_collection):
        assert service.update_user("not-an-id", UserUpdate(first_name="Grace")) is None


class TestUpdatePassword:
    def test_succeeds_when_the_old_password_matches(self, service, users_collection):
        users_collection.find_one.return_value = user_doc(
            password=service.hash_password("old-password")
        )
        assert service.update_password(
            str(OID),
            PasswordUpdate(old_password="old-password", new_password="new-password"),
        )

    def test_writes_a_hash_of_the_new_password(self, service, users_collection):
        users_collection.find_one.return_value = user_doc(
            password=service.hash_password("old-password")
        )
        service.update_password(
            str(OID),
            PasswordUpdate(old_password="old-password", new_password="new-password"),
        )
        stored = users_collection.update_one.call_args[0][1]["$set"]["password"]
        assert stored.startswith("$2b$")
        assert service.verify_password("new-password", stored)

    def test_fails_when_the_old_password_is_wrong(self, service, users_collection):
        users_collection.find_one.return_value = user_doc(
            password=service.hash_password("old-password")
        )
        assert not service.update_password(
            str(OID),
            PasswordUpdate(old_password="wrong-password", new_password="new-password"),
        )

    def test_nothing_is_written_when_the_old_password_is_wrong(
        self, service, users_collection
    ):
        users_collection.find_one.return_value = user_doc(
            password=service.hash_password("old-password")
        )
        service.update_password(
            str(OID),
            PasswordUpdate(old_password="wrong-password", new_password="new-password"),
        )
        users_collection.update_one.assert_not_called()

    def test_fails_when_the_user_is_absent(self, service, users_collection):
        users_collection.find_one.return_value = None
        assert not service.update_password(
            str(OID),
            PasswordUpdate(old_password="old-password", new_password="new-password"),
        )

    def test_fails_for_a_malformed_id(self, service, users_collection):
        assert not service.update_password(
            "not-an-id",
            PasswordUpdate(old_password="old-password", new_password="new-password"),
        )
