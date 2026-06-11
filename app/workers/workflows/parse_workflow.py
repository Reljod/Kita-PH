from hatchet_sdk import Context
from app.workers.hatchet import hatchet
from app.dependencies.services import get_services
from app.models.event import ParseInput

@hatchet.task(name="parse-file", on_events=["file:completed"], input_validator=ParseInput)
async def parse_file_task(input: ParseInput, ctx: Context):
    file_id = input.file_id
    org_id = input.org_id
    
    if not file_id or not org_id:
        return {"status": "error", "message": "Missing file_id or org_id"}

    # Retrieve services from ServiceRegistry singleton
    services = get_services(org_id)

    try:
        await services.parse_service.parse_file(file_id, org_id)
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        print(f"Error parsing file {file_id}: {str(e)}")
        return {"status": "failed", "error": str(e)}
