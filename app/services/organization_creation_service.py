import os
from datetime import datetime, timezone
from typing import Optional
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
    
    def generate_context_llm(self) -> str:
        context_model = os.getenv("CONTEXT_LLM_MODEL", "google/gemini-2.0-flash-001")
        name = f"openrouter/{context_model}"
        from app.models.llm import LlmCreateRequest
        req = LlmCreateRequest(
            name=name,
            model=context_model,
            provider="openrouter"
        )
        llm = self.llm_service.add_llm(req)
        return llm.id

    async def generate_default_agents(self, llm_id: str):
        from bson import ObjectId
        from pathlib import Path

        # Helper to read instructions from markdown files
        base_dir = Path(__file__).parent / "agents"
        
        def read_instructions(subdir: str, fallback: str) -> str:
            path = base_dir / subdir / "instructions.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
            return fallback

        kita_backstory = read_instructions(
            "kita_assistant_agent",
            "You are a state-of-the-art AI assistant, designed to be professional, helpful, and efficient. You have access to tools that enhance your capabilities."
        )
        creator_backstory = read_instructions(
            "creator_agent",
            "You are an expert AI Agent Creator. Your role is to design and formulate specialized AI agents based on user requests. You create agents of different roles, goals, and backstories."
        )

        # Helper to resolve registered tool database ObjectIDs by name
        async def get_tool_id(name: str) -> Optional[str]:
            t = await self.tool_service.get_tool_by_name(name)
            return str(t["_id"]) if t else None

        # 1. Kita Assistant Tools
        kita_tools = []
        rag_search_id = await get_tool_id("rag_search")
        if rag_search_id:
            kita_tools.append(rag_search_id)
        search_mem_id = await get_tool_id("search_memory")
        if search_mem_id:
            kita_tools.append(search_mem_id)
        delegate_id = await get_tool_id("delegate_task")
        if delegate_id:
            kita_tools.append(delegate_id)

        # 2. Agent Creator Tools
        creator_tool_names = [
            "create_agent", 
            "get_agent", 
            "list_agents", 
            "update_agent", 
            "rag_search", 
            "list_available_llms",
            "delegate_task"
        ]
        creator_tools = []
        for name in creator_tool_names:
            tid = await get_tool_id(name)
            if tid:
                creator_tools.append(tid)

        # 3. Context Agent LLM and tools
        context_llm_id = self.generate_context_llm()
        context_backstory = read_instructions(
            "context_agent",
            "You are a context extraction agent. Extract important facts from conversations."
        )
        context_tools = []
        remember_fact_id = await get_tool_id("remember_fact")
        if remember_fact_id:
            context_tools.append(remember_fact_id)

        # Seeding agents
        kita_id = ObjectId()
        kita_doc = {
            "_id": kita_id,
            "base_id": str(kita_id),
            "version": 1,
            "name": "Kita Assistant",
            "role": "Helpful AI Assistant",
            "goal": "Provide concise and expert assistance with any task or question.",
            "backstory": kita_backstory,
            "personalities": ["Helpful", "Professional", "Concise"],
            "llm_id": llm_id,
            "tools": kita_tools,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        creator_id = ObjectId()
        creator_doc = {
            "_id": creator_id,
            "base_id": str(creator_id),
            "version": 1,
            "name": "Agent Creator",
            "role": "Expert AI Agent Creator",
            "goal": "Design and formulate specialized AI agents based on user requests.",
            "backstory": creator_backstory,
            "personalities": None,
            "llm_id": llm_id,
            "tools": creator_tools,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }

        context_id = ObjectId()
        context_doc = {
            "_id": context_id,
            "base_id": str(context_id),
            "version": 1,
            "name": "__system__context_agent",
            "role": "Context Extraction Agent",
            "goal": "Extract and store important facts from conversations to provide personalized responses.",
            "backstory": context_backstory,
            "personalities": None,
            "llm_id": context_llm_id,
            "tools": context_tools,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }


        self.agent_service.collection.insert_one(kita_doc)
        self.agent_service.collection.insert_one(creator_doc)
        self.agent_service.collection.insert_one(context_doc)


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
        org_id = self.agent_service.collection.org_id
        print(f"[REVERT] Reverting organization initialization for {org_id} due to failure.")
        self.llm_service.collection.delete_many({})
        self.agent_service.collection.delete_many({})
        self.tool_service.collection.delete_many({})
        self.rag_service.collection.delete_many({})
        from app.db import db
        db.get_chat_contexts_collection().delete_many({"org_id": org_id})
        print(f"[REVERT] Successfully deleted all scaffolding records for {org_id}.")

