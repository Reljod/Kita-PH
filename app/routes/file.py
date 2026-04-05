from fastapi import APIRouter, HTTPException, Depends, status, Query, Header
from typing import List, Optional
from app.models.file import FileResponse, FileUploadRequest, FileUploadResponse, FileUpdateRequest, BatchFileCompleteRequest
from app.services.file_service import FileService
from app.services.event_service import IEventService, HatchetEventService
from app.db import db, TenantCollection
from app.security import get_current_org_id

router = APIRouter(prefix="/files", tags=["Files"])

def get_event_service() -> IEventService:
    return HatchetEventService()

def get_file_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id"),
    event_service: IEventService = Depends(get_event_service)
) -> FileService:
    collection = TenantCollection(db.get_files_collection(), org_id)
    return FileService(collection, org_id, event_service, agent_id=x_agent_id)

@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def initiate_upload(
    req: FileUploadRequest,
    file_service: FileService = Depends(get_file_service)
):
    try:
        return await file_service.initiate_upload(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating upload: {str(e)}")

@router.get("", response_model=List[FileResponse])
async def list_files(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    file_service: FileService = Depends(get_file_service)
):
    try:
        return await file_service.get_files(agent_id=agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@router.get("/{file_id}", response_model=FileResponse)
async def get_file(
    file_id: str,
    file_service: FileService = Depends(get_file_service)
):
    file = await file_service.get_file(file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@router.patch("/{file_id}", response_model=FileResponse)
async def update_file(
    file_id: str,
    req: FileUpdateRequest,
    file_service: FileService = Depends(get_file_service)
):
    file = await file_service.update_file(file_id, req.model_dump(exclude_unset=True))
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    file_service: FileService = Depends(get_file_service)
):
    success = await file_service.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "File deleted successfully"}

@router.post("/{file_id}/complete", response_model=FileResponse)
async def complete_upload(
    file_id: str,
    file_service: FileService = Depends(get_file_service)
):
    file = await file_service.complete_upload(file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@router.post("/batch-complete", response_model=list[FileResponse])
async def batch_complete_uploads(
    req: BatchFileCompleteRequest,
    file_service: FileService = Depends(get_file_service)
):
    try:
        return await file_service.batch_complete_uploads(req.file_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error completing batch uploads: {str(e)}")
