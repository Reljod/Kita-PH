from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.llm import LlmCreateRequest, LlmResponse
from app.services.llm_service import LlmService, ILlmService

router = APIRouter(prefix="/llm", tags=["LLM Management"])

def get_llm_service() -> ILlmService:
    return LlmService()

@router.post("/", response_model=LlmResponse)
def add_llm(req: LlmCreateRequest, service: ILlmService = Depends(get_llm_service)):
    return service.add_llm(req)

@router.get("/", response_model=List[LlmResponse])
def list_llms(service: ILlmService = Depends(get_llm_service)):
    return service.list_llms()

@router.delete("/{llm_id}")
def delete_llm(llm_id: str, service: ILlmService = Depends(get_llm_service)):
    try:
        success = service.delete_llm(llm_id)
        if not success:
            raise HTTPException(status_code=404, detail="LLM not found")
        return {"message": "LLM deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
