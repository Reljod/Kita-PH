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
                print(f"Registering missing available tools for org: {org_id}")
                for t_name, t_desc in available.items():
                    existing_t = tools_coll.find_one({"org_id": org_id, "name": t_name})
                    if not existing_t:
                        tools_coll.insert_one({
                            "org_id": org_id,
                            "name": t_name,
                            "description": t_desc,
                            "created_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc)
                        })
                        print(f"Registered tool '{t_name}' for Org '{org_id}'")
                
                # Fetch delegate_tool again after registration
                delegate_tool = tools_coll.find_one({"org_id": org_id, "name": "delegate_task"})
                
            if not delegate_tool:
                print(f"Error: Still could not find 'delegate_task' tool for organization: {org_id}")
                continue
                
            tool_id_str = str(delegate_tool["_id"])
            current_tools = agent.get("tools") or []
            
            # Check if already has it
            if tool_id_str not in current_tools:
                # Add to tools array
                agents_coll.update_one(
                    {"_id": agent["_id"]},
                    {"$addToSet": {"tools": tool_id_str}}
                )
                print(f"Added 'delegate_task' to agent '{agent.get('name')}' (Org: {org_id}).")
                updated_count += 1
            else:
                print(f"Agent '{agent.get('name')}' already has 'delegate_task'.")
                
        print(f"Backfill complete. Updated {updated_count} agents.")
    finally:
        db.close()

if __name__ == "__main__":
    backfill()
