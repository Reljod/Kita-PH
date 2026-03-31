import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import create_client, Client
from app.db import TenantCollection
from app.models.file import FileDocument, FileUploadRequest, FileUploadResponse, FileResponse

class FileService:
    def __init__(self, collection: TenantCollection, org_id: str, agent_id: Optional[str] = None):
        self.collection = collection
        self.org_id = org_id
        self.agent_id = agent_id
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and (SUPABASE_SECRET_KEY or SUPABASE_KEY) must be set")
            
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
        
        # Create DB record
        file_doc = FileDocument(
            id=file_id,
            filename=req.filename,
            extension=extension,
            size=req.size,
            content_type=req.content_type,
            org_id=self.org_id,
            agent_id=req.agent_id or self.agent_id,
            metadata=req.metadata or {}
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
        target_agent_id = agent_id or self.agent_id
        
        if target_agent_id:
            # Show agent-specific files AND organization-wide files
            query["$or"] = [{"agent_id": target_agent_id}, {"agent_id": None}]
        else:
            # Show ONLY organization-wide files
            query["agent_id"] = None
            
        docs = self.collection.find(query).sort("created_at", -1)
        return [self._format_response(doc) for doc in docs]

    async def get_file(self, file_id: str) -> Optional[FileResponse]:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            return None
        return self._format_response(doc)

    async def delete_file(self, file_id: str) -> bool:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            return False
            
        # Delete from DB
        self.collection.delete_one({"id": file_id})
        
        # Delete from Supabase
        extension = doc.get("extension", "")
        storage_path = f"{file_id}.{extension}" if extension else file_id
        try:
            self.supabase.storage.from_(self.bucket_name).remove([storage_path])
        except Exception as e:
            print(f"Error deleting file from Supabase: {e}")
            
        return True

    async def update_file(self, file_id: str, req: Dict[str, Any]) -> Optional[FileResponse]:
        doc = self.collection.find_one({"id": file_id})
        if not doc:
            return None
            
        update_data = {k: v for k, v in req.items() if v is not None}
        if not update_data:
            return self._format_response(doc)
            
        update_data["updated_at"] = datetime.now(timezone.utc) if hasattr(datetime, "now") else datetime.utcnow()
        
        self.collection.update_one({"id": file_id}, {"$set": update_data})
        
        updated_doc = self.collection.find_one({"id": file_id})
        return self._format_response(updated_doc)

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
            created_at=doc["created_at"],
            updated_at=doc["updated_at"]
        )
