import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory of this script to sys.path to resolve 'app' imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Load environment variables, prioritizing .env.local
load_dotenv(project_root / ".env.local")
load_dotenv(project_root / ".env")

from app.db import db
from bson import ObjectId
from datetime import datetime, timezone
from app.services.tools import get_available_tools

def backfill():
    db.connect()
    try:
        agents_coll = db.db["agents"]
        tools_coll = db.db["tools"]
        
        agents = list(agents_coll.find({}))
        print(f"Found {len(agents)} agents in the database.")
        
        available = get_available_tools()
        
        updated_count = 0
        for agent in agents:
            org_id = agent.get("org_id")
            if not org_id:
                print(f"Skipping agent '{agent.get('name')}' (no org_id).")
                continue
                
            # Ensure delegate_task is registered for this org
            delegate_tool = tools_coll.find_one({"org_id": org_id, "name": "delegate_task"})
            if not delegate_tool:
                # Only register the delegate_task tool — stay within backfill scope
                if "delegate_task" in available:
                    print(f"Registering 'delegate_task' tool for org: {org_id}")
                    tools_coll.insert_one({
                        "org_id": org_id,
                        "name": "delegate_task",
                        "description": available["delegate_task"],
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    })
                    print(f"Registered tool 'delegate_task' for Org '{org_id}'")
                
                # Fetch delegate_tool again after registration
                delegate_tool = tools_coll.find_one({"org_id": org_id, "name": "delegate_task"})
                
            if not delegate_tool:
                print(f"Error: Still could not find 'delegate_task' tool for organization: {org_id}")
                continue
                
            tool_id_str = str(delegate_tool["_id"])
            current_tools = agent.get("tools") or []
            
            # Check if already has it
            if tool_id_str not in current_tools:
                # Insert a new version document to preserve immutable version history
                latest_agent = agents_coll.find_one(
                    {"base_id": agent.get("base_id", str(agent["_id"]))},
                    sort=[("version", -1)]
                ) or agent
                new_version_num = latest_agent.get("version", 1) + 1
                new_tools = list(latest_agent.get("tools") or [])
                if tool_id_str not in new_tools:
                    new_tools.append(tool_id_str)
                new_doc = {k: v for k, v in latest_agent.items() if k != "_id"}
                new_doc["tools"] = new_tools
                new_doc["version"] = new_version_num
                new_doc["updated_at"] = datetime.now(timezone.utc)
                agents_coll.insert_one(new_doc)
                print(f"Added 'delegate_task' to agent '{agent.get('name')}' (Org: {org_id}) as version {new_version_num}.")
                updated_count += 1
            else:
                print(f"Agent '{agent.get('name')}' already has 'delegate_task'.")
                
        print(f"Backfill complete. Updated {updated_count} agents.")
    finally:
        db.close()

if __name__ == "__main__":
    backfill()
