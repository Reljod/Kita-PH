from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.rag import RagResponse
from app.services.rag.retrieval_service import IRetrievalService
from app.dependencies import get_retrieval_service

router = APIRouter(prefix="/rag", tags=["RAG"])

class RagSearchRequest(BaseModel):
    query: str = Field(..., description="The search query.")
    limit: Optional[int] = Field(5, description="The maximum number of results to return.")

@router.post("/search", response_model=List[RagResponse], status_code=status.HTTP_200_OK)
async def search_rag(
    req: RagSearchRequest,
    retrieval_service: IRetrievalService = Depends(get_retrieval_service)
):
    """
    Execute Structured RAG Search using the Ultimate structured RAG pipeline (RetrievalService).
    """
    try:
        return await retrieval_service.search(query=req.query, limit=req.limit or 5)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error executing RAG search: {str(e)}"
        )
