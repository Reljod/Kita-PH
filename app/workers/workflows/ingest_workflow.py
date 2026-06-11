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

    # Get file metadata to construct the path/reference for the agent
    file_res = await services.file_service.get_file(file_id)
    if not file_res:
        return {"status": "error", "message": f"File {file_id} not found in metadata service"}
    
    file_path = f"{file_id}.{file_res.extension}"
    
    # Run the Rag Manager Agent
    agent = services.agent_service.get_runnable_agent("rag-manager")
    
    deps = {
        "org_id": org_id,
        "agent_id": "rag-manager",
        "agent_service": services.agent_service,
        "file_service": services.file_service,
        "parse_service": services.parse_service,
        "graph_rag_service": services.graph_rag_service
    }
    
    prompt = f"Please ingest the file '{file_path}' into the Graph RAG system."
    
    try:
        # We run the agent directly. In the future, we might want to record this as a 'System Chat'.
        await agent.run(prompt, deps=deps)
        return {"status": "success", "file_id": file_id, "message": "Ingestion completed"}
    except Exception as e:
        print(f"Error ingesting file {file_id}: {str(e)}")
        return {"status": "failed", "error": str(e)}
