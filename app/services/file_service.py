import os
import uuid
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from supabase import create_client, Client
from app.db import TenantCollection
from app.models.file import FileDocument, FileUploadRequest, FileUploadResponse, FileResponse, FileStatus
from app.services.event_service import IEventService

from app.exceptions import (
    SystemConfigurationError,
    KitaFileNotFoundError,
    FileUploadFailedError
)

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self, collection: TenantCollection, org_id: str, event_service: IEventService):
        self.collection = collection
        self.org_id = org_id
        self.event_service = event_service
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise SystemConfigurationError("SUPABASE_URL and (SUPABASE_SECRET_KEY or SUPABASE_KEY) must be set in environment variables")
            
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.bucket_name = "files"

    def _get_extension(self, filename: str) -> str:
        if "." in filename:
            return filename.split(".")[-1].lower()
        return ""

    async def initiate_upload(self, req: FileUploadRequest) -> FileUploadResponse:
        file_id = str(uuid.uuid4())
        extension = self._get_extension(req.filename)
        storage_path = f"{file_id}.{extension}" if extension else file_id
        
        logger.info(
            f"Initiating file upload: filename={req.filename}, size={req.size} bytes, file_id={file_id}",
            extra={"filename": req.filename, "size": req.size, "file_id": file_id}
        )

        # Create DB record
        file_doc = FileDocument(
            id=file_id,
            filename=req.filename,
            extension=extension,
            size=req.size,
            content_type=req.content_type,
            org_id=self.org_id,
            agent_id=req.agent_id,
            metadata=req.metadata or {},
            status=FileStatus.PENDING
        )
        
        self.collection.insert_one(file_doc.model_dump())
        
        # Determine upload method
        # Standard upload for < 6MB, Resumable for >= 6MB
        # However, the user wants signed URLs for both or specifically for resumable.
        # "Use objectId + extension to create pre-signed url for upload using supabase, 
        # resumable upload and standard upload (For less then 6mb)."
        
        if req.size < 6 * 1024 * 1024:
            # Standard signed upload URL
            res = self.supabase.storage.from_(self.bucket_name).create_signed_upload_url(storage_path)
            return FileUploadResponse(
                file_id=file_id,
                upload_url=res["signed_url"],
                method="POST",
                token=res.get("token")
            )
        else:
            # Resumable signed upload URL
            # Note: create_signed_upload_url returns a token that can be used for resumable uploads
            # in the x-signature header.
            res = self.supabase.storage.from_(self.bucket_name).create_signed_upload_url(storage_path)
            
            # For TUS/Resumable, we use the direct storage hostname
            project_id = os.getenv("SUPABASE_URL", "").split("//")[1].split(".")[0]
            tus_endpoint = f"https://{project_id}.storage.supabase.co/storage/v1/upload/resumable"
            
            return FileUploadResponse(
                file_id=file_id,
                upload_url=tus_endpoint,
                method="TUS",
                token=res.get("token")
            )

    async def get_files(self, agent_id: Optional[str] = None) -> List[FileResponse]:
        query = {}
        if agent_id:
            # Show agent-specific files AND organization-wide files
            query["$or"] = [{"agent_id": agent_id}, {"agent_id": None}]
        else:
            # Show ONLY organization-wide files
            query["agent_id"] = None
            
        docs = self.collection.find(query).sort("created_at", -1)
        return [self._format_response(doc) for doc in docs]

    async def get_file(self, file_id: str) -> FileResponse:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            raise KitaFileNotFoundError(file_id)
        return self._format_response(doc)

    async def delete_file(self, file_id: str) -> bool:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            raise KitaFileNotFoundError(file_id)
            
        logger.info(
            f"Deleting file: filename={doc.get('filename')}, file_id={file_id}",
            extra={"filename": doc.get('filename'), "file_id": file_id}
        )

        # Delete from DB
        self.collection.delete_one({"id": file_id})
        
        # Delete from Supabase
        extension = doc.get("extension", "")
        storage_path = f"{file_id}.{extension}" if extension else file_id
        try:
            self.supabase.storage.from_(self.bucket_name).remove([storage_path])
        except Exception as e:
            # We log but continue, ensuring DB record is cleaned up 
            # even if storage is out of sync or file already gone.
            logger.error(f"Error deleting file from Supabase storage: {e}", exc_info=True)
            
        return True

    async def update_file(self, file_id: str, req: Dict[str, Any]) -> FileResponse:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            raise KitaFileNotFoundError(file_id)
            
        update_data = {k: v for k, v in req.items() if v is not None}
        if not update_data:
            return self._format_response(doc)
            
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        self.collection.update_one({"id": file_id}, {"$set": update_data})
        
        updated_doc = self.collection.find_one({"id": file_id})
        return self._format_response(updated_doc)

    async def complete_upload(self, file_id: str) -> FileResponse:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            raise KitaFileNotFoundError(file_id)
            
        update_data = {
            "status": FileStatus.COMPLETED,
            "updated_at": datetime.now(timezone.utc)
        }
        
        self.collection.update_one({"id": file_id}, {"$set": update_data})
        
        updated_doc = self.collection.find_one({"id": file_id})
        res = self._format_response(updated_doc)
        
        logger.info(
            f"Completed file upload: filename={res.filename}, file_id={res.id}",
            extra={"filename": res.filename, "file_id": res.id}
        )

        # Trigger event
        await self.event_service.push("file:completed", {
            "file_id": res.id,
            "filename": res.filename,
            "org_id": res.org_id,
            "agent_id": res.agent_id,
            "extension": res.extension,
            "size": res.size,
            "content_type": res.content_type,
            "metadata": res.metadata
        })
        
        return res

    async def batch_complete_uploads(self, file_ids: List[str]) -> List[FileResponse]:
        results = []
        for file_id in file_ids:
            try:
                res = await self.complete_upload(file_id)
                results.append(res)
            except KitaFileNotFoundError:
                # Skip files not found during batch completion to be robust
                continue
        return results

    async def download_file(self, file_id: str) -> bytes:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            raise KitaFileNotFoundError(file_id)
            
        logger.info(
            f"Downloading file: filename={doc.get('filename')}, file_id={file_id}",
            extra={"filename": doc.get('filename'), "file_id": file_id}
        )

        extension = doc.get("extension", "")
        storage_path = f"{file_id}.{extension}" if extension else file_id
        
        try:
            return await asyncio.to_thread(
                self.supabase.storage.from_(self.bucket_name).download,
                storage_path
            )
        except Exception as e:
            raise FileUploadFailedError(f"Failed to download file from storage: {str(e)}")

    def _format_response(self, doc: Dict[str, Any]) -> FileResponse:
        return FileResponse(
            id=doc["id"],
            filename=doc["filename"],
            extension=doc["extension"],
            size=doc["size"],
            content_type=doc.get("content_type"),
            org_id=doc["org_id"],
            agent_id=doc.get("agent_id"),
            metadata=doc.get("metadata"),
            status=doc.get("status", FileStatus.PENDING),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"]
        )
