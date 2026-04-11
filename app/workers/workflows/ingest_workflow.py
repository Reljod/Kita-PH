from hatchet_sdk import Context
from app.workers.hatchet import hatchet
from app.services.agent_service import AgentService
from app.services.file_service import FileService
from app.services.parse_service import LlamaParseService
from app.services.graph_rag_service import Neo4JGraphRagService
from app.services.event_service import HatchetEventService
from app.services.llm_service import LlmService
from app.db import db, TenantCollection
import os
from app.models.event import IngestInput

@hatchet.task(name="ingest-file", on_events=["parse:completed"], input_validator=IngestInput)
async def ingest_file_task(input: IngestInput, ctx: Context):
    file_id = input.file_id
    org_id = input.org_id
    
    if not file_id or not org_id:
        return {"status": "error", "message": "Missing file_id or org_id"}

    # Initialize services
    event_service = HatchetEventService()
    
    # Repos and Services for Dependencies
    llm_coll = TenantCollection(db.get_llms_collection(), org_id)
    llm_service = LlmService(llm_coll)
    
    agent_coll = TenantCollection(db.get_agents_collection(), org_id)
    tools_coll = TenantCollection(db.get_tools_collection(), org_id)
    agent_service = AgentService(llm_service, agent_coll, tools_coll)
    
    file_coll = TenantCollection(db.get_files_collection(), org_id)
    file_service = FileService(file_coll, org_id, event_service)
    
    parse_coll = TenantCollection(db.get_file_parse_collection(), org_id)
    parse_service = LlamaParseService(parse_coll, file_service, event_service)
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    graph_service = Neo4JGraphRagService(neo4j_uri, neo4j_user, neo4j_password, org_id)

    # Get file metadata to construct the path/reference for the agent
    file_res = await file_service.get_file(file_id)
    if not file_res:
        return {"status": "error", "message": f"File {file_id} not found in metadata service"}
    
    file_path = f"{file_id}.{file_res.extension}"
    
    # Run the Rag Manager Agent
    agent = agent_service.get_runnable_agent("rag-manager")
    
    deps = {
        "org_id": org_id,
        "agent_id": "rag-manager",
        "agent_service": agent_service,
        "file_service": file_service,
        "parse_service": parse_service,
        "graph_rag_service": graph_service
    }
    
    prompt = f"Please ingest the file '{file_path}' into the Graph RAG system."
    
    try:
        # We run the agent directly. In the future, we might want to record this as a 'System Chat'.
        await agent.run(prompt, deps=deps)
        return {"status": "success", "file_id": file_id, "message": "Ingestion completed"}
    except Exception as e:
        print(f"Error ingesting file {file_id}: {str(e)}")
        return {"status": "failed", "error": str(e)}
