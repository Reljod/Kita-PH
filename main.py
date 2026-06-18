import os
import warnings

# Disable gRPC fork handlers to suppress fork warnings on MacOS
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

# Suppress deprecation warnings from libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hatchet_sdk")

import logging
import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv


# Load environment variables, prioritizing .env.local if it exists
load_dotenv(".env.local")
load_dotenv()

from app.db import db
from app.routes import chat, memory, agent, llm, auth, user, organization, tool, file, event, rag
from app.routes.webhook import facebook
from app.security import require_org_membership
from fastapi import Depends
from app.services.redis_service import RedisService
from app.middleware.error_handler import setup_error_handlers
from app.middleware.api_key_auth import ApiKeyAuthMiddleware
from app.utils.logger import setup_logging, CorrelationIdMiddleware, get_global_headers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB and Redis
    db.connect()
    try:
        redis_client = RedisService.get_client()
        await redis_client.ping()
        print("Successfully connected to Redis.")
    except Exception as e:
        print(f"Failed to connect to Redis during startup: {e}")
    yield
    # Shutdown: Close connections
    db.close()
    await RedisService.close()

app = FastAPI(
    title="Kita API", 
    description="LLM Python FastAPI app with pymongo and pydantic-ai",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(get_global_headers)]
)

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
if log_level_str not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    log_level_str = "INFO"

app_env = os.getenv("APP_ENV", "local").lower()

logfire.configure(
    environment=app_env,
    distributed_tracing=False,
    scrubbing=False,
    console=logfire.ConsoleOptions(
        min_log_level=log_level_str.lower(),
        span_style='indented',
        include_timestamps=True,
    )
)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()
logfire.instrument_fastapi(app)

# Initialize standard logging with custom LogFormatter
setup_logging()

# Configure CORS
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://kita-ph-ui.vercel.app"
]
origins_env = os.environ.get("CORS_ALLOWED_ORIGINS")
if origins_env:
    allowed_origins.extend([origin.strip() for origin in origins_env.split(",") if origin.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://kita-ph-.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(ApiKeyAuthMiddleware)

# Set up global exception handlers
setup_error_handlers(app)



# Include Routers
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(organization.router)
app.include_router(facebook.router)

# Protected Routers - Require Organization Membership
protected_deps = [Depends(require_org_membership)]
app.include_router(chat.router)
app.include_router(memory.router, dependencies=protected_deps)
app.include_router(llm.router, dependencies=protected_deps)
app.include_router(agent.router, dependencies=protected_deps)
app.include_router(tool.router, dependencies=protected_deps)
app.include_router(file.router, dependencies=protected_deps)
app.include_router(event.router, dependencies=protected_deps)
app.include_router(rag.router, dependencies=protected_deps)

@app.get("/")
def root():
    return {
        "message": "Welcome to Kita API", 
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    if log_level_str not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level_str = "INFO"
    reload_mode = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes") or log_level_str == "DEBUG"
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        log_level=log_level_str.lower(),
        reload=reload_mode
    )
