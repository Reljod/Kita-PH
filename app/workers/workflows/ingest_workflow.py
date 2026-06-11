from datetime import timedelta
from hatchet_sdk import Context
from app.workers.hatchet import hatchet
from app.dependencies.services import get_services
from app.models.event import IngestInput

@hatchet.task(
    name="ingest-file", 
    on_events=["parse:completed"], 
    input_validator=IngestInput,
    execution_timeout=timedelta(minutes=10)
)
async def ingest_file_task(input: IngestInput, ctx: Context):
    file_id = input.file_id
    org_id = input.org_id
    
    if not file_id or not org_id:
        return {"status": "error", "message": "Missing file_id or org_id"}

    # Retrieve services from ServiceRegistry singleton
    services = get_services(org_id)
    
    try:
        # Direct ingestion using MongoDB IngestService
        success = await services.ingest_service.ingest_file_parse(file_id, org_id)
        if success:
            return {"status": "success", "file_id": file_id, "message": "Ingestion completed"}
        else:
            return {"status": "failed", "error": "Ingestion service returned failure"}
    except Exception as e:
        print(f"Error ingesting file {file_id}: {str(e)}")
        return {"status": "failed", "error": str(e)}
