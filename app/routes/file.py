import logging
from fastapi import APIRouter, HTTPException, Depends, status, Query, Header
from typing import List, Optional
from app.models.file import FileResponse, FileUploadRequest, FileUploadResponse, FileUpdateRequest, BatchFileCompleteRequest
from app.services.file_service import FileService
from app.services.event_service import IEventService, HatchetEventService
from app.db import db, TenantCollection
from app.security import get_current_org_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["Files"])

from app.dependencies import get_file_service

@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def initiate_upload(
    req: FileUploadRequest,
    file_service: FileService = Depends(get_file_service)
):
    logger.info(f"Initiating upload for file: {req.filename} (size: {req.file_size_bytes} bytes)")
    try:
        res = await file_service.initiate_upload(req)
        logger.info(f"Successfully initiated upload. Assigned file_id: {res.file_id}")
        return res
    except Exception as e:
        logger.error(f"Failed to initiate upload for file {req.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initiating upload: {str(e)}")

@router.get("", response_model=List[FileResponse])
async def list_files(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    file_service: FileService = Depends(get_file_service)
):
    try:
        return await file_service.get_files(agent_id=agent_id)
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
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
    logger.info(f"Updating file attributes for file_id: {file_id}")
    file = await file_service.update_file(file_id, req.model_dump(exclude_unset=True))
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    file_service: FileService = Depends(get_file_service)
):
    logger.info(f"Attempting to delete file_id: {file_id}")
    success = await file_service.delete_file(file_id)
    if not success:
        logger.warning(f"Failed to delete file: file_id {file_id} not found")
        raise HTTPException(status_code=404, detail="File not found")
    logger.info(f"Successfully deleted file_id: {file_id}")
    return {"message": "File deleted successfully"}

@router.post("/{file_id}/complete", response_model=FileResponse)
async def complete_upload(
    file_id: str,
    file_service: FileService = Depends(get_file_service)
):
    logger.info(f"Completing upload for file_id: {file_id}")
    file = await file_service.complete_upload(file_id)
    if not file:
        logger.warning(f"Failed to complete upload: file_id {file_id} not found")
        raise HTTPException(status_code=404, detail="File not found")
    logger.info(f"Successfully completed upload for file_id: {file_id}")
    return file

@router.post("/batch-complete", response_model=list[FileResponse])
async def batch_complete_uploads(
    req: BatchFileCompleteRequest,
    file_service: FileService = Depends(get_file_service)
):
    logger.info(f"Completing batch uploads for file_ids: {req.file_ids}")
    try:
        res = await file_service.batch_complete_uploads(req.file_ids)
        logger.info(f"Successfully completed batch uploads for {len(req.file_ids)} files")
        return res
    except Exception as e:
        logger.error(f"Error completing batch uploads: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error completing batch uploads: {str(e)}")
