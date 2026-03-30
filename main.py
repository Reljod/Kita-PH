import logfire
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic_ai import Agent

from app.db import db
from app.routes import chat, memory, agent, llm, auth, user, organization, tool, file
from app.routes.webhook import facebook
from app.security import require_org_membership
from fastapi import Depends

# Load environment variables, prioritizing .env.local if it exists
load_dotenv(".env.local")
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    db.connect()
    yield
    # Shutdown: Close MongoDB connection
    db.close()

app = FastAPI(
    title="Kita API", 
    description="LLM Python FastAPI app with pymongo and pydantic-ai",
    version="1.0.0",
    lifespan=lifespan
)

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_fastapi(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(organization.router)
app.include_router(facebook.router)

# Protected Routers - Require Organization Membership
protected_deps = [Depends(require_org_membership)]
app.include_router(chat.router, dependencies=protected_deps)
app.include_router(memory.router, dependencies=protected_deps)
app.include_router(llm.router, dependencies=protected_deps)
app.include_router(agent.router, dependencies=protected_deps)
app.include_router(tool.router, dependencies=protected_deps)
app.include_router(file.router, dependencies=protected_deps)

@app.get("/")
def root():
    return {
        "message": "Welcome to Kita API", 
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
