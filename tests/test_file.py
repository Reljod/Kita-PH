import unittest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.middleware.error_handler import setup_error_handlers
from app.models.file import FileUploadResponse, FileResponse, FileStatus


class TestFileRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_app = FastAPI()
        from app.routes.file import router
        cls.test_app.include_router(router)
        setup_error_handlers(cls.test_app)

        cls.mock_file_service = MagicMock()
        from app.dependencies import get_file_service
        cls.test_app.dependency_overrides[get_file_service] = lambda: cls.mock_file_service
        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_initiate_upload_success(self):
        self.mock_file_service.initiate_upload = AsyncMock(return_value=FileUploadResponse(
            file_id="file-123",
            upload_url="https://storage.example.com/upload/file-123",
            method="POST",
            token=None,
        ))
        response = self.client.post("/files/upload", json={
            "filename": "doc.pdf",
            "size": 1024,
            "content_type": "application/pdf",
        })
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["file_id"], "file-123")
        self.assertIn("upload_url", data)

    def test_initiate_upload_validation_error(self):
        response = self.client.post("/files/upload", json={"filename": "", "size": 0})
        self.assertEqual(response.status_code, 422)

    def test_list_files_success(self):
        mock_file = FileResponse(
            id="file-123", filename="doc.pdf", extension=".pdf",
            size=1024, content_type="application/pdf",
            org_id="org-123", agent_id=None, metadata=None,
            status=FileStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.mock_file_service.get_files = AsyncMock(return_value=[mock_file])
        response = self.client.get("/files")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "file-123")

    def test_get_file_success(self):
        mock_file = FileResponse(
            id="file-123", filename="doc.pdf", extension=".pdf",
            size=1024, content_type="application/pdf",
            org_id="org-123", agent_id=None, metadata=None,
            status=FileStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.mock_file_service.get_file = AsyncMock(return_value=mock_file)
        response = self.client.get("/files/file-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "file-123")

    def test_get_file_not_found(self):
        self.mock_file_service.get_file = AsyncMock(return_value=None)
        response = self.client.get("/files/file-999")
        self.assertEqual(response.status_code, 404)

    def test_update_file_success(self):
        mock_file = FileResponse(
            id="file-123", filename="renamed.pdf", extension=".pdf",
            size=1024, content_type="application/pdf",
            org_id="org-123", agent_id=None, metadata=None,
            status=FileStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.mock_file_service.update_file = AsyncMock(return_value=mock_file)
        response = self.client.patch("/files/file-123", json={"filename": "renamed.pdf"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["filename"], "renamed.pdf")

    def test_update_file_not_found(self):
        self.mock_file_service.update_file = AsyncMock(return_value=None)
        response = self.client.patch("/files/file-999", json={"filename": "renamed.pdf"})
        self.assertEqual(response.status_code, 404)

    def test_delete_file_success(self):
        self.mock_file_service.delete_file = AsyncMock(return_value=True)
        response = self.client.delete("/files/file-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "File deleted successfully")

    def test_delete_file_not_found(self):
        self.mock_file_service.delete_file = AsyncMock(return_value=False)
        response = self.client.delete("/files/file-999")
        self.assertEqual(response.status_code, 404)

    def test_complete_upload_success(self):
        mock_file = FileResponse(
            id="file-123", filename="doc.pdf", extension=".pdf",
            size=1024, content_type="application/pdf",
            org_id="org-123", agent_id=None, metadata=None,
            status=FileStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.mock_file_service.complete_upload = AsyncMock(return_value=mock_file)
        response = self.client.post("/files/file-123/complete")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "file-123")

    def test_complete_upload_not_found(self):
        self.mock_file_service.complete_upload = AsyncMock(return_value=None)
        response = self.client.post("/files/file-999/complete")
        self.assertEqual(response.status_code, 404)

    def test_batch_complete_success(self):
        mock_file = FileResponse(
            id="file-123", filename="doc.pdf", extension=".pdf",
            size=1024, content_type="application/pdf",
            org_id="org-123", agent_id=None, metadata=None,
            status=FileStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.mock_file_service.batch_complete_uploads = AsyncMock(return_value=[mock_file])
        response = self.client.post("/files/batch-complete", json={"file_ids": ["file-123"]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["id"], "file-123")

    def test_batch_complete_validation_error(self):
        response = self.client.post("/files/batch-complete", json={"file_ids": []})
        self.assertEqual(response.status_code, 422)
