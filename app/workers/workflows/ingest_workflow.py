from datetime import timedelta
import logging
from hatchet_sdk import Context
from app.workers.hatchet import hatchet
from app.dependencies.services import get_services
from app.models.event import IngestInput

logger = logging.getLogger(__name__)

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
        logger.warning(f"ingest-file task started with missing parameters: file_id={file_id}, org_id={org_id}")
        return {"status": "error", "message": "Missing file_id or org_id"}

    logger.info(
        f"Starting ingest-file task: file_id={file_id}, org_id={org_id}",
        extra={"file_id": file_id, "org_id": org_id}
    )

    # Retrieve services from ServiceRegistry singleton
    services = get_services(org_id)
    
    try:
        # Direct ingestion using MongoDB IngestService
        success = await services.ingest_service.ingest_file_parse(file_id, org_id)
        if success:
            logger.info(
                f"Successfully completed ingest-file task: file_id={file_id}",
                extra={"file_id": file_id, "org_id": org_id}
            )
            return {"status": "success", "file_id": file_id, "message": "Ingestion completed"}
        else:
            logger.error(
                f"Ingestion service returned failure for file_id={file_id}",
                extra={"file_id": file_id, "org_id": org_id}
            )
            return {"status": "failed", "error": "Ingestion service returned failure"}
    except Exception as e:
        logger.error(
            f"Error ingesting file {file_id}: {str(e)}",
            extra={"file_id": file_id, "org_id": org_id, "error": str(e)},
            exc_info=True
        )
        return {"status": "failed", "error": str(e)}
