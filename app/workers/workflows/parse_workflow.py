import logging
from hatchet_sdk import Context
from app.workers.hatchet import hatchet
from app.dependencies.services import get_services
from app.models.event import ParseInput

logger = logging.getLogger(__name__)

@hatchet.task(name="parse-file", on_events=["file:completed"], input_validator=ParseInput)
async def parse_file_task(input: ParseInput, ctx: Context):
    file_id = input.file_id
    org_id = input.org_id
    
    if not file_id or not org_id:
        logger.warning(f"parse-file task started with missing parameters: file_id={file_id}, org_id={org_id}")
        return {"status": "error", "message": "Missing file_id or org_id"}

    logger.info(
        f"Starting parse-file task: file_id={file_id}, org_id={org_id}",
        extra={"file_id": file_id, "org_id": org_id}
    )

    # Retrieve services from ServiceRegistry singleton
    services = get_services(org_id)

    try:
        await services.parse_service.parse_file(file_id, org_id)
        logger.info(
            f"Successfully completed parse-file task: file_id={file_id}",
            extra={"file_id": file_id, "org_id": org_id}
        )
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        logger.error(
            f"Error parsing file {file_id}: {str(e)}",
            extra={"file_id": file_id, "org_id": org_id, "error": str(e)},
            exc_info=True
        )
        return {"status": "failed", "error": str(e)}
