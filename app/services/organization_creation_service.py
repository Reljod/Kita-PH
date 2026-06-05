import os
from datetime import datetime
from app.services.llm_service import ILlmService
from app.services.agent_service import IAgentService
from app.services.tool_service import IToolService
from app.services.rag_service import IRagService
from app.services.organization_service import OrganizationService

class OrganizationCreationService:
    def __init__(
        self,
        llm_service: ILlmService,
        agent_service: IAgentService,
        tool_service: IToolService,
        rag_service: IRagService,
        org_service: OrganizationService
    ):
        self.llm_service = llm_service
        self.agent_service = agent_service
        self.tool_service = tool_service
        self.rag_service = rag_service
        self.org_service = org_service

    async def initialize_org(self, org_id: str):
        try:
            # 1. Generate default LLM
            llm_id = self.generate_default_llm()
            
            # 2. Register all available tools
            await self.generate_default_tools()
            
            # 3. Seeding default agents (Kita Assistant and Agent Creator)
            await self.generate_default_agents(llm_id)
            
            # 4. Initialize global memory with organization details
            await self.initialize_default_global_memories(org_id)
            
            # 5. Set status to completed
            self.org_service.update_org_status(org_id, "completed")
            
        except Exception as e:
            # Mark status as failed
            self.org_service.update_org_status(org_id, "failed")
            # Revert any changes to keep database clean
            try:
                self.revert_initialization()
            except Exception as revert_err:
                print(f"Failed to revert initialization for {org_id}: {revert_err}")
            raise e
    
    def generate_default_llm(self) -> str:
        """ Generate default LLM from environment.

        Make sure it's like this:
        {
            "name": "openrouter/x-ai/grok-4.3",
            "model": "x-ai/grok-4.3",
            "provider": "openrouter"
        }
        """
        model = os.getenv("LLM_MODEL", "x-ai/grok-4.3")
        provider = "openrouter"
        name = f"openrouter/{model}" if not model.startswith("openrouter/") else model
        
        if model.startswith("openrouter/"):
            model = model.replace("openrouter/", "")

        from app.models.llm import LlmCreateRequest
        req = LlmCreateRequest(
            name=name,
            model=model,
            provider=provider
        )
        llm = self.llm_service.add_llm(req)
        return llm.id
    
    async def generate_default_agents(self, llm_id: str):
        search_mem = await self.tool_service.get_tool_by_name("search_memory")
        search_mem_v2 = await self.tool_service.get_tool_by_name("search_memory_v2")
        kita_tools = []
        if search_mem:
            kita_tools.append(str(search_mem["_id"]))
        if search_mem_v2:
            kita_tools.append(str(search_mem_v2["_id"]))

        # 1. Kita Assistant
        kita_doc = {
            "_id": "kita-assistant",
            "base_id": "kita-assistant",
            "version": 1,
            "name": "Kita Assistant",
            "role": "Helpful AI Assistant",
            "goal": "Provide concise and expert assistance with any task or question.",
            "backstory": "You are a state-of-the-art AI assistant, designed to be professional, helpful, and efficient. You have access to tools that enhance your capabilities.",
            "personalities": ["Helpful", "Professional", "Concise"],
            "llm_id": llm_id,
            "tools": kita_tools,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # 2. Agent Creator
        creator_doc = {
            "_id": "agent-creator",
            "base_id": "agent-creator",
            "version": 1,
            "name": "Agent Creator",
            "role": "Expert AI Agent Creator",
            "goal": "Design and formulate specialized AI agents based on user requests.",
            "backstory": "You are an expert AI Agent Creator. Your role is to design and formulate specialized AI agents based on user requests. You create agents of different roles, goals, and backstories.",
            "personalities": None,
            "llm_id": llm_id,
            "tools": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Save default agents to DB
        self.agent_service.collection.insert_one(kita_doc)
        self.agent_service.collection.insert_one(creator_doc)

    async def generate_default_tools(self):
        from app.services.tools import get_available_tools
        available_tools = get_available_tools()
        for name in available_tools.keys():
            await self.tool_service.register_tool(name)

    async def initialize_default_global_memories(self, org_id: str):
        org = self.org_service.get_org(org_id)
        if org:
            from app.models.rag import RagCreateRequest
            req = RagCreateRequest(
                title="Organization Details",
                content=f"Organization Name: {org.org_name}\nOrganization Code: {org.org_code}",
                agent_id=None
            )
            rag_resp = await self.rag_service.add_rag(req)
            # Trigger embedding generation synchronously in the creation pipeline
            await self.rag_service.update_embedding(rag_resp.id)
    
    def revert_initialization(self):
        # Tenant collection automatically scopes queries/deletions to the organization
        self.llm_service.collection.delete_many({})
        self.agent_service.collection.delete_many({})
        self.tool_service.collection.delete_many({})
        self.rag_service.collection.delete_many({})
