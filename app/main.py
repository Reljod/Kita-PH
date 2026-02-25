from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from langsmith.integrations.otel import configure
from pydantic_ai import Agent

from app.db import db
from app.routes import chat, memory, agent, llm, auth, user, organization
from app.security import get_current_user
from fastapi import Depends

# Load environment variables, prioritizing .env.local if it exists
load_dotenv(".env.local")
load_dotenv()

# Configure LangSmith tracing
configure(project_name=os.getenv("LANGSMITH_PROJECT", "kita_ph"))

# Instrument all PydanticAI agents
Agent.instrument_all()

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

# Include Routers
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(organization.router)

# Protected Routers
app.include_router(chat.router, dependencies=[Depends(get_current_user)])
app.include_router(memory.router, dependencies=[Depends(get_current_user)])
app.include_router(llm.router, dependencies=[Depends(get_current_user)])
app.include_router(agent.router, dependencies=[Depends(get_current_user)])

@app.get("/")
def root():
    return {
        "message": "Welcome to Kita API", 
        "docs_url": "/docs"
    }
