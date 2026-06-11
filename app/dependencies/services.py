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


# --- Service Registry / Singleton Container ---

class ServiceRegistry:
    def __init__(self, org_id: str):
        self.org_id = org_id

        # Global singletons
        self.event_service = get_event_service()
        self.facebook_service = get_facebook_service()
        self.org_service = get_org_service()
        self.web_search_service = get_web_search_service()

        # Tenant-scoped services
        self.llm_service = LlmService(TenantCollection(db.get_llms_collection(), org_id))
        
        self.agent_service = AgentService(
            llm_service=self.llm_service,
            collection=TenantCollection(db.get_agents_collection(), org_id),
            tools_collection=TenantCollection(db.get_tools_collection(), org_id)
        )
        
        self.tool_service = ToolService(
            web_search_service=self.web_search_service,
            collection=TenantCollection(db.get_tools_collection(), org_id)
        )
        
        self.file_service = FileService(
            collection=TenantCollection(db.get_files_collection(), org_id),
            org_id=org_id,
            event_service=self.event_service
        )
        
        self.parse_service = LlamaParseService(
            parse_collection=TenantCollection(db.get_file_parse_collection(), org_id),
            file_service=self.file_service,
            event_service=self.event_service
        )
        
        import os
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.graph_rag_service = Neo4JGraphRagService(neo4j_uri, neo4j_user, neo4j_password, org_id)
        
        self.rag_service = MongoVectorDbRagService(
            collection=TenantCollection(db.get_rag_collection(), org_id)
        )
        
        self.chat_service = ChatService(
            agent_service=self.agent_service,
            collection=TenantCollection(db.get_chats_collection(), org_id),
            file_service=self.file_service,
            parse_service=self.parse_service,
            graph_rag_service=self.graph_rag_service,
            rag_service=self.rag_service
        )


_registries: dict[str, ServiceRegistry] = {}


def get_services(org_id: str) -> ServiceRegistry:
    if org_id not in _registries:
        _registries[org_id] = ServiceRegistry(org_id)
    return _registries[org_id]


# --- Scoped / Tenant Dependencies ---

def get_llm_service(org_id: str = Depends(get_current_org_id)) -> ILlmService:
    return get_services(org_id).llm_service


def get_agent_service(
    org_id: str = Depends(get_current_org_id)
) -> IAgentService:
    return get_services(org_id).agent_service


def get_tool_service(
    org_id: str = Depends(get_current_org_id)
) -> IToolService:
    return get_services(org_id).tool_service


def get_file_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
) -> FileService:
    return get_services(org_id).file_service


def get_parse_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
) -> LlamaParseService:
    return get_services(org_id).parse_service


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
    return get_services(org_id).graph_rag_service


def get_rag_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
) -> IRagService:
    return get_services(org_id).rag_service


def get_agent_rag_service(
    agent_id: str,
    org_id: str = Depends(get_current_org_id)
) -> IRagService:
    return get_services(org_id).rag_service


def get_chat_service(
    org_id: str = Depends(get_current_org_id),
    x_agent_id: Optional[str] = Header(None, alias="x-agent-id")
) -> IChatService:
    return get_services(org_id).chat_service


