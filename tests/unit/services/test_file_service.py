"""Tests for app.services.file_service.

Uploads are two-phase: this service hands back a signed destination, the
client pushes bytes straight to Supabase, and a later call flips the record to
completed. Nothing here ever sees the bytes, so the behaviour worth pinning is
the record lifecycle, the 6MB standard/resumable split, and what happens when
storage and the database disagree.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db import TenantCollection
from app.exceptions import (
    FileUploadFailedError,
    KitaFileNotFoundError,
    SystemConfigurationError,
)
from app.models.file import FileStatus, FileUploadRequest
from app.services.file_service import FileService

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"
AGENT_ID = "agent_1"
SMALL = 1024
LARGE = 7 * 1024 * 1024


@pytest.fixture
def files_collection(mongo_db):
    return mongo_db["files"]


@pytest.fixture
def storage() -> MagicMock:
    bucket = MagicMock(name="bucket")
    bucket.create_signed_upload_url.return_value = {
        "signed_url": "https://storage.example/signed",
        "token": "signed-token",
    }
    bucket.remove.return_value = None
    bucket.download.return_value = b"file bytes"
    return bucket


@pytest.fixture
def supabase(monkeypatch, storage) -> MagicMock:
    client = MagicMock(name="supabase")
    client.storage.from_.return_value = storage
    monkeypatch.setattr(
        "app.services.file_service.create_client", lambda url, key: client
    )
    return client


@pytest.fixture
def event_service() -> MagicMock:
    service = MagicMock(name="event_service")
    service.push = AsyncMock()
    return service


@pytest.fixture
def service(supabase, files_collection, event_service) -> FileService:
    return FileService(
        TenantCollection(files_collection, ORG_ID), ORG_ID, event_service
    )


def a_request(**overrides) -> FileUploadRequest:
    payload = {"filename": "report.pdf", "size": SMALL}
    payload.update(overrides)
    return FileUploadRequest(**payload)


# --- construction ---------------------------------------------------------


class TestConstruction:
    def test_missing_storage_configuration_is_a_configuration_error(
        self, monkeypatch, files_collection, event_service
    ):
        """Surfacing this as a 500 at request time would blame the caller for
        a deployment mistake."""
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        with pytest.raises(SystemConfigurationError):
            FileService(
                TenantCollection(files_collection, ORG_ID), ORG_ID, event_service
            )

    def test_a_missing_key_is_also_a_configuration_error(
        self, monkeypatch, files_collection, event_service
    ):
        monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        with pytest.raises(SystemConfigurationError):
            FileService(
                TenantCollection(files_collection, ORG_ID), ORG_ID, event_service
            )

    def test_the_secret_key_is_preferred_over_the_anon_key(
        self, monkeypatch, files_collection, event_service
    ):
        """The anon key cannot write to a private bucket, so falling back to
        it silently would break uploads rather than fail loudly."""
        seen = {}

        def record(url, key):
            seen["key"] = key
            return MagicMock()

        monkeypatch.setattr("app.services.file_service.create_client", record)
        monkeypatch.setenv("SUPABASE_SECRET_KEY", "the-secret")
        monkeypatch.setenv("SUPABASE_KEY", "the-anon")
        FileService(TenantCollection(files_collection, ORG_ID), ORG_ID, event_service)
        assert seen["key"] == "the-secret"


# --- initiating an upload -------------------------------------------------


class TestInitiateUpload:
    async def test_a_destination_is_returned(self, service):
        assert (await service.initiate_upload(a_request())).upload_url

    async def test_a_record_is_written(self, service, files_collection):
        response = await service.initiate_upload(a_request())
        assert files_collection.find_one({"id": response.file_id})

    async def test_the_record_starts_pending(self, service, files_collection):
        """Anything downstream keys off completed, so a record that started
        there would be parsed before its bytes existed."""
        response = await service.initiate_upload(a_request())
        stored = files_collection.find_one({"id": response.file_id})
        assert stored["status"] == FileStatus.PENDING

    async def test_the_record_is_scoped_to_the_organization(
        self, service, files_collection
    ):
        response = await service.initiate_upload(a_request())
        assert files_collection.find_one({"id": response.file_id})["org_id"] == ORG_ID

    async def test_the_extension_is_taken_from_the_filename(
        self, service, files_collection
    ):
        response = await service.initiate_upload(a_request(filename="notes.MD"))
        stored = files_collection.find_one({"id": response.file_id})
        assert stored["extension"] == "md"

    async def test_a_filename_without_an_extension_is_accepted(
        self, service, files_collection
    ):
        response = await service.initiate_upload(a_request(filename="LICENSE"))
        stored = files_collection.find_one({"id": response.file_id})
        assert stored["extension"] == ""

    async def test_only_the_last_dot_separates_the_extension(
        self, service, files_collection
    ):
        response = await service.initiate_upload(a_request(filename="a.tar.gz"))
        stored = files_collection.find_one({"id": response.file_id})
        assert stored["extension"] == "gz"

    async def test_the_storage_path_carries_the_extension(self, service, storage):
        """Supabase serves by path, so a stored object with no extension gets
        the wrong content type on download."""
        response = await service.initiate_upload(a_request(filename="notes.md"))
        assert storage.create_signed_upload_url.call_args[0][0] == (
            f"{response.file_id}.md"
        )

    async def test_each_upload_gets_its_own_id(self, service):
        first = await service.initiate_upload(a_request())
        second = await service.initiate_upload(a_request())
        assert first.file_id != second.file_id

    async def test_an_agent_can_be_attached(self, service, files_collection):
        response = await service.initiate_upload(a_request(agent_id=AGENT_ID))
        stored = files_collection.find_one({"id": response.file_id})
        assert stored["agent_id"] == AGENT_ID

    async def test_metadata_defaults_to_empty(self, service, files_collection):
        response = await service.initiate_upload(a_request())
        assert files_collection.find_one({"id": response.file_id})["metadata"] == {}


class TestUploadMethodSplit:
    async def test_a_small_file_uses_a_direct_post(self, service):
        assert (await service.initiate_upload(a_request(size=SMALL))).method == "POST"

    async def test_a_large_file_uses_a_resumable_upload(self, service):
        """A dropped connection partway through a 40MB upload should not mean
        starting over."""
        assert (await service.initiate_upload(a_request(size=LARGE))).method == "TUS"

    async def test_the_boundary_is_six_megabytes(self, service):
        just_under = await service.initiate_upload(a_request(size=6 * 1024 * 1024 - 1))
        exactly = await service.initiate_upload(a_request(size=6 * 1024 * 1024))
        assert just_under.method == "POST" and exactly.method == "TUS"

    async def test_a_resumable_upload_points_at_the_tus_endpoint(self, service):
        response = await service.initiate_upload(a_request(size=LARGE))
        assert response.upload_url.endswith("/upload/resumable")

    async def test_both_methods_return_a_token(self, service):
        small = await service.initiate_upload(a_request(size=SMALL))
        large = await service.initiate_upload(a_request(size=LARGE))
        assert small.token and large.token


# --- reads ----------------------------------------------------------------


class TestGetFiles:
    async def test_no_files_yields_an_empty_list(self, service):
        assert await service.get_files() == []

    async def test_organization_wide_files_are_listed(self, service):
        await service.initiate_upload(a_request())
        assert len(await service.get_files()) == 1

    async def test_agent_files_are_hidden_from_the_organization_view(self, service):
        """The unfiltered view is the organization's shared library, not
        every file anyone ever attached to an agent."""
        await service.initiate_upload(a_request(agent_id=AGENT_ID))
        assert await service.get_files() == []

    async def test_the_agent_view_includes_shared_files(self, service):
        """An agent can use the organization's library as well as its own."""
        await service.initiate_upload(a_request())
        await service.initiate_upload(a_request(agent_id=AGENT_ID))
        assert len(await service.get_files(agent_id=AGENT_ID)) == 2

    async def test_another_agents_files_are_not_listed(self, service):
        await service.initiate_upload(a_request(agent_id="agent_other"))
        assert await service.get_files(agent_id=AGENT_ID) == []

    async def test_another_organizations_files_are_not_listed(
        self, service, files_collection, event_service, supabase
    ):
        await service.initiate_upload(a_request())
        intruder = FileService(
            TenantCollection(files_collection, OTHER_ORG), OTHER_ORG, event_service
        )
        assert await intruder.get_files() == []


class TestGetFile:
    async def test_a_file_is_returned(self, service):
        response = await service.initiate_upload(a_request())
        assert (await service.get_file(response.file_id)).id == response.file_id

    async def test_a_missing_file_raises(self, service):
        with pytest.raises(KitaFileNotFoundError):
            await service.get_file("nope")

    async def test_another_organization_cannot_read_the_file(
        self, service, files_collection, event_service, supabase
    ):
        response = await service.initiate_upload(a_request())
        intruder = FileService(
            TenantCollection(files_collection, OTHER_ORG), OTHER_ORG, event_service
        )
        with pytest.raises(KitaFileNotFoundError):
            await intruder.get_file(response.file_id)


# --- updates --------------------------------------------------------------


class TestUpdateFile:
    async def test_a_field_is_updated(self, service):
        response = await service.initiate_upload(a_request())
        updated = await service.update_file(response.file_id, {"filename": "new.pdf"})
        assert updated.filename == "new.pdf"

    async def test_none_values_are_ignored(self, service):
        """A partial update sends the whole model with unset fields as None;
        writing those through would blank the record."""
        response = await service.initiate_upload(a_request())
        updated = await service.update_file(response.file_id, {"filename": None})
        assert updated.filename == "report.pdf"

    async def test_an_empty_update_returns_the_record_unchanged(self, service):
        response = await service.initiate_upload(a_request())
        updated = await service.update_file(response.file_id, {})
        assert updated.filename == "report.pdf"

    async def test_a_missing_file_raises(self, service):
        with pytest.raises(KitaFileNotFoundError):
            await service.update_file("nope", {"filename": "x"})

    async def test_the_timestamp_moves(self, service):
        response = await service.initiate_upload(a_request())
        before = (await service.get_file(response.file_id)).updated_at
        updated = await service.update_file(response.file_id, {"filename": "new.pdf"})
        assert updated.updated_at >= before


# --- completion -----------------------------------------------------------


class TestCompleteUpload:
    async def test_the_file_becomes_completed(self, service):
        response = await service.initiate_upload(a_request())
        assert (
            await service.complete_upload(response.file_id)
        ).status == FileStatus.COMPLETED

    async def test_a_missing_file_raises(self, service):
        with pytest.raises(KitaFileNotFoundError):
            await service.complete_upload("nope")

    async def test_the_parse_pipeline_is_notified(self, service, event_service):
        """Completion is the only signal parsing and RAG get; without it an
        uploaded file is never indexed."""
        response = await service.initiate_upload(a_request())
        await service.complete_upload(response.file_id)
        assert event_service.push.await_args[0][0] == "file:completed"

    async def test_the_event_carries_what_the_worker_needs(
        self, service, event_service
    ):
        response = await service.initiate_upload(a_request(agent_id=AGENT_ID))
        await service.complete_upload(response.file_id)
        payload = event_service.push.await_args[0][1]
        assert payload["file_id"] == response.file_id
        assert payload["org_id"] == ORG_ID
        assert payload["agent_id"] == AGENT_ID

    async def test_completing_twice_is_harmless(self, service):
        response = await service.initiate_upload(a_request())
        await service.complete_upload(response.file_id)
        assert (
            await service.complete_upload(response.file_id)
        ).status == FileStatus.COMPLETED


class TestBatchCompleteUploads:
    async def test_every_file_is_completed(self, service):
        first = await service.initiate_upload(a_request())
        second = await service.initiate_upload(a_request())
        results = await service.batch_complete_uploads([first.file_id, second.file_id])
        assert len(results) == 2

    async def test_a_missing_file_does_not_abort_the_batch(self, service):
        """A client retrying a partially-applied batch should not be blocked
        by the ids that already went through."""
        existing = await service.initiate_upload(a_request())
        results = await service.batch_complete_uploads(["nope", existing.file_id])
        assert [r.id for r in results] == [existing.file_id]

    async def test_an_empty_batch_yields_nothing(self, service):
        assert await service.batch_complete_uploads([]) == []

    async def test_only_the_found_files_raise_events(self, service, event_service):
        await service.batch_complete_uploads(["nope"])
        event_service.push.assert_not_awaited()


# --- deletion -------------------------------------------------------------


class TestDeleteFile:
    async def test_the_record_is_removed(self, service, files_collection):
        response = await service.initiate_upload(a_request())
        await service.delete_file(response.file_id)
        assert files_collection.count_documents({}) == 0

    async def test_the_stored_object_is_removed(self, service, storage):
        response = await service.initiate_upload(a_request())
        await service.delete_file(response.file_id)
        storage.remove.assert_called_once_with([f"{response.file_id}.pdf"])

    async def test_a_missing_file_raises(self, service):
        with pytest.raises(KitaFileNotFoundError):
            await service.delete_file("nope")

    async def test_a_storage_failure_still_removes_the_record(
        self, service, storage, files_collection
    ):
        """An object already gone from storage would otherwise leave a record
        that can never be deleted."""
        storage.remove.side_effect = RuntimeError("storage unavailable")
        response = await service.initiate_upload(a_request())
        assert await service.delete_file(response.file_id) is True
        assert files_collection.count_documents({}) == 0

    async def test_another_organization_cannot_delete_the_file(
        self, service, files_collection, event_service, supabase
    ):
        response = await service.initiate_upload(a_request())
        intruder = FileService(
            TenantCollection(files_collection, OTHER_ORG), OTHER_ORG, event_service
        )
        with pytest.raises(KitaFileNotFoundError):
            await intruder.delete_file(response.file_id)
        assert files_collection.count_documents({}) == 1


# --- download -------------------------------------------------------------


class TestDownloadFile:
    async def test_the_bytes_are_returned(self, service):
        response = await service.initiate_upload(a_request())
        assert await service.download_file(response.file_id) == b"file bytes"

    async def test_the_stored_path_is_used(self, service, storage):
        response = await service.initiate_upload(a_request())
        await service.download_file(response.file_id)
        storage.download.assert_called_once_with(f"{response.file_id}.pdf")

    async def test_a_missing_record_raises(self, service):
        with pytest.raises(KitaFileNotFoundError):
            await service.download_file("nope")

    async def test_a_storage_failure_is_reported(self, service, storage):
        storage.download.side_effect = RuntimeError("object missing")
        response = await service.initiate_upload(a_request())
        with pytest.raises(FileUploadFailedError):
            await service.download_file(response.file_id)

    async def test_a_file_without_an_extension_downloads_by_bare_id(
        self, service, storage
    ):
        response = await service.initiate_upload(a_request(filename="LICENSE"))
        await service.download_file(response.file_id)
        storage.download.assert_called_once_with(response.file_id)
