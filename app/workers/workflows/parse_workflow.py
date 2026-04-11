from hatchet_sdk import Context
from app.workers.hatchet import hatchet
from app.services.parse_service import LlamaParseService
from app.services.file_service import FileService
from app.services.event_service import HatchetEventService
from app.db import db, TenantCollection
import os

from pydantic import BaseModel

class ParseInput(BaseModel):
    file_id: str
    org_id: str

@hatchet.task(name="parse-file", on_events=["file:completed"], input_validator=ParseInput)
async def parse_file_task(input: ParseInput, ctx: Context):
    file_id = input.file_id
    org_id = input.org_id
    
    if not file_id or not org_id:
        return {"status": "error", "message": "Missing file_id or org_id"}

    # Initialize services
    # No need to initialize twice on some services
    event_service = HatchetEventService()
    file_service = FileService(
        collection=TenantCollection(db.get_files_collection(), org_id),
        org_id=org_id,
        event_service=event_service
    )
    
    parse_service = LlamaParseService(
        parse_collection=TenantCollection(db.get_file_parse_collection(), org_id),
        file_service=file_service,
        event_service=event_service
    )

    try:
        await parse_service.parse_file(file_id, org_id)
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        print(f"Error parsing file {file_id}: {str(e)}")
        return {"status": "failed", "error": str(e)}
