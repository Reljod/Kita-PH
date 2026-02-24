from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import List
from app.models.rag import RagCreateRequest, RagResponse, RagUpdateRequest
from app.services.rag_service import MongoVectorDbRagService, IRagService

router = APIRouter(prefix="/memory", tags=["Memory"])

def get_rag_service() -> IRagService:
    return MongoVectorDbRagService()

@router.post("", response_model=RagResponse, status_code=status.HTTP_201_CREATED)
async def create_rag(
    req: RagCreateRequest, 
    background_tasks: BackgroundTasks,
    rag_service: IRagService = Depends(get_rag_service)
):
    try:
        rag = await rag_service.add_rag(req)
        # Trigger background embedding
        background_tasks.add_task(rag_service.update_embedding, rag.id)
        return rag
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating memory: {str(e)}")

@router.get("", response_model=List[RagResponse])
async def get_all_rags(rag_service: IRagService = Depends(get_rag_service)):
    return rag_service.get_all_rags()

@router.get("/{rag_id}", response_model=RagResponse)
async def get_rag(rag_id: str, rag_service: IRagService = Depends(get_rag_service)):
    try:
        rag = rag_service.get_rag(rag_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not rag:
        raise HTTPException(status_code=404, detail="Memory not found")
        
    return rag

@router.put("/{rag_id}", response_model=RagResponse)
async def update_rag(
    rag_id: str, 
    req: RagUpdateRequest, 
    background_tasks: BackgroundTasks,
    rag_service: IRagService = Depends(get_rag_service)
):
    try:
        rag = await rag_service.edit_rag(rag_id, req)
        if not rag:
            raise HTTPException(status_code=404, detail="Memory not found")
            
        # If status is pending (which service sets if content updated), trigger background re-embedding
        if rag.status == "pending":
            background_tasks.add_task(rag_service.update_embedding, rag.id)
            
        return rag
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating memory: {str(e)}")

@router.delete("/{rag_id}")
async def delete_rag(rag_id: str, rag_service: IRagService = Depends(get_rag_service)):
    try:
        success = await rag_service.delete_rag(rag_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"message": "Memory deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")
