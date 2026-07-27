"""Tests for app.routes.file — upload initiation, listing, and completion.

Uploads are two-phase: `initiate_upload` hands back a destination, the client
pushes bytes to storage directly, then `complete_upload` flips the record to
completed. The status transitions and the 404s around them are what these
cover; the storage round trip itself belongs to the E2E suite.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.dependencies import get_file_service
from app.models.file import FileResponse, FileStatus, FileUploadResponse
from app.routes import file as file_routes

from .conftest import override

FILE_ID = "file_1"
ORG_ID = "org_abc"
AGENT_ID = "agent_1"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def a_file(**overrides) -> FileResponse:
    payload = {
        "id": FILE_ID,
        "filename": "report.pdf",
        "extension": "pdf",
        "size": 1024,
        "content_type": "application/pdf",
        "org_id": ORG_ID,
        "status": FileStatus.PENDING,
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return FileResponse(**payload)


def an_upload() -> FileUploadResponse:
    return FileUploadResponse(
        file_id=FILE_ID,
        upload_url="https://storage.example/upload/file_1",
        method="POST",
        token="signed-token",
    )


@pytest.fixture
def file_service() -> MagicMock:
    service = MagicMock(name="file_service")
    service.initiate_upload = AsyncMock(return_value=an_upload())
    service.get_files = AsyncMock(return_value=[a_file()])
    service.get_file = AsyncMock(return_value=a_file())
    service.update_file = AsyncMock(return_value=a_file())
    service.delete_file = AsyncMock(return_value=True)
    service.complete_upload = AsyncMock(
        return_value=a_file(status=FileStatus.COMPLETED)
    )
    service.batch_complete_uploads = AsyncMock(
        return_value=[a_file(status=FileStatus.COMPLETED)]
    )
    return service


@pytest.fixture
def client(make_client, file_service):
    return make_client(file_routes.router, {get_file_service: override(file_service)})


def upload_payload(**overrides) -> dict:
    body = {"filename": "report.pdf", "size": 1024}
    body.update(overrides)
    return body


# --- initiate -------------------------------------------------------------


class TestInitiateUpload:
    def test_an_upload_is_initiated(self, client):
        assert client.post("/files/upload", json=upload_payload()).status_code == 201

    def test_the_destination_is_returned(self, client):
        body = client.post("/files/upload", json=upload_payload()).json()
        assert body["file_id"] == FILE_ID
        assert body["upload_url"].startswith("https://")

    def test_the_request_reaches_the_service(self, client, file_service):
        client.post("/files/upload", json=upload_payload(filename="notes.md"))
        assert file_service.initiate_upload.await_args[0][0].filename == "notes.md"

    def test_an_agent_can_be_attached_at_upload_time(self, client, file_service):
        client.post("/files/upload", json=upload_payload(agent_id=AGENT_ID))
        assert file_service.initiate_upload.await_args[0][0].agent_id == AGENT_ID

    def test_a_storage_failure_is_a_500(self, client, file_service):
        file_service.initiate_upload = AsyncMock(side_effect=RuntimeError("s3 down"))
        assert client.post("/files/upload", json=upload_payload()).status_code == 500

    @pytest.mark.parametrize(
        "body",
        [
            {"filename": "", "size": 1024},
            {"filename": "report.pdf", "size": 0},
            {"filename": "report.pdf", "size": -1},
            {"filename": "report.pdf"},
            {"size": 1024},
            {"filename": "x" * 256, "size": 1024},
        ],
    )
    def test_invalid_input_is_rejected(self, client, file_service, body):
        assert client.post("/files/upload", json=body).status_code == 422
        file_service.initiate_upload.assert_not_awaited()

    def test_a_file_over_the_size_cap_is_rejected(self, client, file_service):
        """50MB ceiling — enforced before anything is provisioned in storage."""
        response = client.post("/files/upload", json=upload_payload(size=52428801))
        assert response.status_code == 422
        file_service.initiate_upload.assert_not_awaited()

    def test_a_file_exactly_at_the_cap_is_accepted(self, client):
        response = client.post("/files/upload", json=upload_payload(size=52428800))
        assert response.status_code == 201


# --- read -----------------------------------------------------------------


class TestListFiles:
    def test_files_are_listed(self, client):
        response = client.get("/files")
        assert response.status_code == 200 and len(response.json()) == 1

    def test_no_filter_lists_the_whole_organization(self, client, file_service):
        client.get("/files")
        file_service.get_files.assert_awaited_once_with(agent_id=None)

    def test_the_agent_filter_is_applied(self, client, file_service):
        client.get("/files", params={"agent_id": AGENT_ID})
        file_service.get_files.assert_awaited_once_with(agent_id=AGENT_ID)

    def test_an_empty_result_is_an_empty_list(self, client, file_service):
        file_service.get_files = AsyncMock(return_value=[])
        assert client.get("/files").json() == []

    def test_a_service_failure_is_a_500(self, client, file_service):
        file_service.get_files = AsyncMock(side_effect=RuntimeError("mongo down"))
        assert client.get("/files").status_code == 500


class TestGetFile:
    def test_a_file_is_returned(self, client):
        assert client.get(f"/files/{FILE_ID}").status_code == 200

    def test_a_missing_file_is_a_404(self, client, file_service):
        file_service.get_file = AsyncMock(return_value=None)
        assert client.get(f"/files/{FILE_ID}").status_code == 404


# --- update / delete ------------------------------------------------------


class TestUpdateFile:
    def test_a_file_is_updated(self, client):
        response = client.patch(f"/files/{FILE_ID}", json={"filename": "renamed.pdf"})
        assert response.status_code == 200

    def test_only_the_supplied_fields_are_written(self, client, file_service):
        """exclude_unset matters here: sending the whole model would null out
        metadata and agent_id on every rename."""
        client.patch(f"/files/{FILE_ID}", json={"filename": "renamed.pdf"})
        assert file_service.update_file.await_args[0][1] == {"filename": "renamed.pdf"}

    def test_an_empty_update_writes_nothing(self, client, file_service):
        client.patch(f"/files/{FILE_ID}", json={})
        assert file_service.update_file.await_args[0][1] == {}

    def test_the_status_can_be_set(self, client, file_service):
        client.patch(f"/files/{FILE_ID}", json={"status": "failed"})
        assert file_service.update_file.await_args[0][1]["status"] == FileStatus.FAILED

    def test_an_unknown_status_is_rejected(self, client, file_service):
        response = client.patch(f"/files/{FILE_ID}", json={"status": "nonsense"})
        assert response.status_code == 422
        file_service.update_file.assert_not_awaited()

    def test_a_missing_file_is_a_404(self, client, file_service):
        file_service.update_file = AsyncMock(return_value=None)
        response = client.patch(f"/files/{FILE_ID}", json={"filename": "renamed.pdf"})
        assert response.status_code == 404

    def test_a_blank_filename_is_rejected(self, client):
        assert (
            client.patch(f"/files/{FILE_ID}", json={"filename": ""}).status_code == 422
        )


class TestDeleteFile:
    def test_a_file_is_deleted(self, client):
        assert client.delete(f"/files/{FILE_ID}").status_code == 200

    def test_the_delete_targets_the_path_id(self, client, file_service):
        client.delete(f"/files/{FILE_ID}")
        file_service.delete_file.assert_awaited_once_with(FILE_ID)

    def test_a_missing_file_is_a_404(self, client, file_service):
        file_service.delete_file = AsyncMock(return_value=False)
        assert client.delete(f"/files/{FILE_ID}").status_code == 404


# --- completion -----------------------------------------------------------


class TestCompleteUpload:
    def test_an_upload_is_completed(self, client):
        assert client.post(f"/files/{FILE_ID}/complete").status_code == 200

    def test_the_file_comes_back_completed(self, client):
        """Completion is what makes the file visible to parsing and RAG, so
        the status flip is the point of the call."""
        body = client.post(f"/files/{FILE_ID}/complete").json()
        assert body["status"] == FileStatus.COMPLETED

    def test_a_missing_file_is_a_404(self, client, file_service):
        file_service.complete_upload = AsyncMock(return_value=None)
        assert client.post(f"/files/{FILE_ID}/complete").status_code == 404


class TestBatchCompleteUploads:
    def test_a_batch_is_completed(self, client):
        response = client.post("/files/batch-complete", json={"file_ids": [FILE_ID]})
        assert response.status_code == 200 and len(response.json()) == 1

    def test_the_ids_reach_the_service(self, client, file_service):
        client.post("/files/batch-complete", json={"file_ids": ["a", "b"]})
        file_service.batch_complete_uploads.assert_awaited_once_with(["a", "b"])

    def test_an_empty_batch_is_rejected(self, client, file_service):
        response = client.post("/files/batch-complete", json={"file_ids": []})
        assert response.status_code == 422
        file_service.batch_complete_uploads.assert_not_awaited()

    def test_an_oversized_batch_is_rejected(self, client, file_service):
        response = client.post(
            "/files/batch-complete", json={"file_ids": [f"f{i}" for i in range(101)]}
        )
        assert response.status_code == 422
        file_service.batch_complete_uploads.assert_not_awaited()

    def test_a_batch_of_exactly_the_cap_is_accepted(self, client):
        response = client.post(
            "/files/batch-complete", json={"file_ids": [f"f{i}" for i in range(100)]}
        )
        assert response.status_code == 200

    def test_a_service_failure_is_a_500(self, client, file_service):
        file_service.batch_complete_uploads = AsyncMock(
            side_effect=RuntimeError("storage timeout")
        )
        response = client.post("/files/batch-complete", json={"file_ids": [FILE_ID]})
        assert response.status_code == 500
