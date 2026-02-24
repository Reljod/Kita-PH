from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

from app.db import db
from app.routes import chat

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

# Include Routers
app.include_router(chat.router)

@app.get("/")
def root():
    return {
        "message": "Welcome to Kita API", 
        "docs_url": "/docs"
    }
