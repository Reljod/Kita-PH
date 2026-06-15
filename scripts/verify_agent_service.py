import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory of this script to sys.path to resolve 'app' imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

load_dotenv(project_root / ".env.local")
load_dotenv(project_root / ".env")

from app.db import db, TenantCollection
from app.services.agent_service import AgentService
from app.services.llm_service import LlmService
from app.models.agent import AgentCreateRequest

async def test_versioning():
    db.connect()
    try:
        org_id = "test_org_123"
        agent_coll = TenantCollection(db.get_agents_collection(), org_id)
        tools_coll = TenantCollection(db.get_tools_collection(), org_id)
        llm_coll = TenantCollection(db.get_llms_collection(), org_id)
        
        # Cleanup
        agent_coll.delete_many({})
        tools_coll.delete_many({})
        llm_coll.delete_many({})
        
        # Create LLM
        from app.models.llm import LlmDocument
        llm_doc = LlmDocument(
            name="test-llm",
            model="x-ai/grok-4.3",
            provider="openrouter"
        )
        llm_res = llm_coll.insert_one(llm_doc.model_dump())
        llm_id = str(llm_res.inserted_id)
        
        # Register a couple of dummy tools
        tools_coll.insert_one({"name": "web_search", "description": "web"})
        tools_coll.insert_one({"name": "delegate_task", "description": "delegate"})
        
        web_tool = tools_coll.find_one({"name": "web_search"})
        web_tool_id = str(web_tool["_id"])
        
        delegate_tool = tools_coll.find_one({"name": "delegate_task"})
        delegate_tool_id = str(delegate_tool["_id"])
        
        agent_service = AgentService(
            llm_service=LlmService(llm_coll),
            collection=agent_coll,
            tools_collection=tools_coll
        )
        
        # 1. Create Agent
        req = AgentCreateRequest(
            name="Test Agent",
            role="Tester",
            goal="Test versioning",
            backstory="I test things",
            llm_id=llm_id,
            tools=[]
        )
        agent = await agent_service.create_agent(req)
        print(f"Created agent: {agent.name}, ID: {agent.id}, Version: {agent.version}, Tools: {agent.tools}")
        
        # 2. Add tools
        await agent_service.add_tools(agent.id, [web_tool_id])
        
        # Load agent again
        agent_v2 = agent_service.get_agent(agent.id)
        print(f"After add_tools: {agent_v2.name}, ID: {agent_v2.id}, Version: {agent_v2.version}, Tools: {agent_v2.tools}")
        
        # 3. Add another tool
        await agent_service.add_tools(agent.id, [delegate_tool_id])
        agent_v3 = agent_service.get_agent(agent.id)
        print(f"After second add_tools: {agent_v3.name}, ID: {agent_v3.id}, Version: {agent_v3.version}, Tools: {agent_v3.tools}")
        
        # 4. Remove tool
        await agent_service.remove_tools(agent.id, [web_tool_id])
        agent_v4 = agent_service.get_agent(agent.id)
        print(f"After remove_tools: {agent_v4.name}, ID: {agent_v4.id}, Version: {agent_v4.version}, Tools: {agent_v4.tools}")
        
        # Cleanup test data
        agent_coll.delete_many({})
        tools_coll.delete_many({})
        llm_coll.delete_many({})
        print("Verification completed successfully and test data cleaned up.")
    finally:
        db.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_versioning())
