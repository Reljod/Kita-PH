from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, Query, Header
from typing import List, Optional
from app.models.rag import RagCreateRequest, RagResponse, RagUpdateRequest
from app.models.graph_rag import GraphRagSearchResult
from app.services.rag_service import MongoVectorDbRagService, IRagService
from app.services.graph_rag_service import Neo4JGraphRagService, GraphRagService
from app.db import db, TenantCollection
from app.security import get_current_org_id
import os

router = APIRouter(prefix="/memory", tags=["Memory"])

def get_rag_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
) -> IRagService:
    collection = TenantCollection(db.get_rag_collection(), org_id)
    return MongoVectorDbRagService(collection, agent_id=x_agent_id)

def get_graph_rag_service(
    org_id: str = Depends(get_current_org_id)
) -> GraphRagService:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        raise HTTPException(status_code=500, detail="Graph RAG environment variables are not properly configured.")
    return Neo4JGraphRagService(uri, user, password, org_id)

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

@router.get("/search", response_model=List[RagResponse])
async def search_memory(
    query: str = Query(..., description="The search query to find relevant information in memory."), 
    limit: int = Query(5, description="The maximum number of results to return."), 
    rag_service: IRagService = Depends(get_rag_service)
):
    try:
        return await rag_service.search(query, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memory: {str(e)}")

@router.get("/search/v2", response_model=List[GraphRagSearchResult])
async def search_memory_v2(
    query: str = Query(..., description="The search query to find relevant information in graph RAG memory."), 
    limit: int = Query(5, description="The maximum number of results to return."), 
    agent_id: Optional[str] = Query(None, description="Optional agent ID filter"),
    filename: Optional[str] = Query(None, description="Optional filename filter"),
    graph_rag_service: GraphRagService = Depends(get_graph_rag_service)
):
    """
    Graph-based RAG search with flexible filtering.
    """
    try:
        filters = {}
        if agent_id:
            filters["agent_id"] = agent_id
        if filename:
            filters["filename"] = filename
            
        return await graph_rag_service.query(query, limit=limit, filters=filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching graph memory: {str(e)}")

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
