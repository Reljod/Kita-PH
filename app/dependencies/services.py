from typing import Optional, Any
from fastapi import Depends, Header

# Security / Auth
from app.security import get_current_org_id

# Interfaces / Classes
from app.services.event_service import IEventService, HatchetEventService
from app.services.webhook.facebook_service import FacebookService
from app.services.organization_service import OrganizationService
from app.services.web_search_service import SerperSearchService
from app.services.llm_service import ILlmService, LlmService
from app.services.agent_service import IAgentService, AgentService
from app.services.tool_service import IToolService, ToolService
from app.services.file_service import FileService
from app.services.parse_service import LlamaParseService
from app.services.graph_rag_service import Neo4JGraphRagService
from app.services.rag_service import IRagService, MongoVectorDbRagService
from app.services.chat_service import IChatService, ChatService

# Database
from app.db import db, TenantCollection

# --- Singletons ---
_event_service: Optional[IEventService] = None
_facebook_service: Optional[FacebookService] = None
_org_service: Optional[OrganizationService] = None
_web_search_service: Optional[SerperSearchService] = None


def get_event_service() -> IEventService:
    global _event_service
    if _event_service is None:
        _event_service = HatchetEventService()
    return _event_service


def get_facebook_service() -> FacebookService:
    global _facebook_service
    if _facebook_service is None:
        _facebook_service = FacebookService()
    return _facebook_service


def get_org_service() -> OrganizationService:
    global _org_service
    if _org_service is None:
        _org_service = OrganizationService()
    return _org_service


def get_web_search_service() -> SerperSearchService:
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = SerperSearchService()
    return _web_search_service


# --- Scoped / Tenant Dependencies ---

def get_llm_service(org_id: str = Depends(get_current_org_id)) -> ILlmService:
    collection = TenantCollection(db.get_llms_collection(), org_id)
    return LlmService(collection)


def get_agent_service(
    org_id: str = Depends(get_current_org_id),
    llm_service: ILlmService = Depends(get_llm_service)
) -> IAgentService:
    collection = TenantCollection(db.get_agents_collection(), org_id)
    tools_collection = TenantCollection(db.get_tools_collection(), org_id)
    return AgentService(llm_service=llm_service, collection=collection, tools_collection=tools_collection)


def get_tool_service(
    org_id: str = Depends(get_current_org_id),
    web_search_service: SerperSearchService = Depends(get_web_search_service)
) -> IToolService:
    collection = TenantCollection(db.get_tools_collection(), org_id)
    return ToolService(web_search_service=web_search_service, collection=collection)


def get_file_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id"),
    event_service: IEventService = Depends(get_event_service)
) -> FileService:
    collection = TenantCollection(db.get_files_collection(), org_id)
    return FileService(collection, org_id, event_service, agent_id=x_agent_id)


def get_parse_service(
    org_id: str = Depends(get_current_org_id),
    file_service: FileService = Depends(get_file_service),
    event_service: IEventService = Depends(get_event_service)
) -> LlamaParseService:
    parse_coll = TenantCollection(db.get_file_parse_collection(), org_id)
    return LlamaParseService(parse_coll, file_service, event_service)


def get_graph_rag_service(
    org_id: str = Depends(get_current_org_id)
) -> Neo4JGraphRagService:
    import os
    from fastapi import HTTPException
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise HTTPException(status_code=500, detail="Graph RAG environment variables are not properly configured.")
    return Neo4JGraphRagService(neo4j_uri, neo4j_user, neo4j_password, org_id)


def get_rag_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
) -> IRagService:
    collection = TenantCollection(db.get_rag_collection(), org_id)
    return MongoVectorDbRagService(collection, agent_id=x_agent_id)


def get_agent_rag_service(
    agent_id: str,
    org_id: str = Depends(get_current_org_id)
) -> IRagService:
    collection = TenantCollection(db.get_rag_collection(), org_id)
    return MongoVectorDbRagService(collection, agent_id=agent_id)


def get_chat_service(
    org_id: str = Depends(get_current_org_id),
    agent_service: IAgentService = Depends(get_agent_service),
    file_service: FileService = Depends(get_file_service),
    parse_service: LlamaParseService = Depends(get_parse_service),
    graph_rag_service: Neo4JGraphRagService = Depends(get_graph_rag_service),
    rag_service: IRagService = Depends(get_rag_service)
) -> IChatService:
    collection = TenantCollection(db.get_chats_collection(), org_id)
    return ChatService(
        agent_service=agent_service,
        collection=collection,
        file_service=file_service,
        parse_service=parse_service,
        graph_rag_service=graph_rag_service,
        rag_service=rag_service
    )
