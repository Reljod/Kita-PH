from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
from app.models.event import EventPushRequest, EventKey, ParseInput, IngestInput
from app.services.event_service import IEventService, HatchetEventService
from pydantic import ValidationError

router = APIRouter(prefix="/events", tags=["Events"])

def get_event_service() -> IEventService:
    return HatchetEventService()

@router.post("/push", status_code=status.HTTP_200_OK)
async def push_event(
    req: EventPushRequest,
    event_service: IEventService = Depends(get_event_service)
):
    """
    Push a custom event to Hatchet with validation based on the event key.
    """
    # Validation logic based on event key
    try:
        if req.event_key == EventKey.FILE_COMPLETED:
            # Validates that the payload contains required fields for parse-file task
            ParseInput(**req.payload)
        elif req.event_key == EventKey.PARSE_COMPLETED:
            # Validates that the payload contains required fields for ingest-file task
            IngestInput(**req.payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=e.errors()
        )
    
    # Push event
    try:
        await event_service.push(req.event_key, req.payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pushing event: {str(e)}")
        
    return {"message": "Event pushed successfully", "event": req.event_key}
