import os
import warnings

# Disable gRPC fork handlers to suppress fork warnings on MacOS
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

# Suppress deprecation warnings from libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hatchet_sdk")

import logfire
from dotenv import load_dotenv
from app.db import db
from app.workers.hatchet import hatchet
from app.workers.workflows.first_workflow import my_task
from app.workers.workflows.parse_workflow import parse_file_task
from app.workers.workflows.ingest_workflow import ingest_file_task
from app.utils.logger import setup_logging

def main():
    # Load environment variables
    load_dotenv(".env.local")
    load_dotenv()
    
    # Configure Logfire
    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_openai()
    
    # Initialize standard logging with custom LogFormatter
    setup_logging()
    
    # Initialize database connection
    db.connect()
    
    worker = hatchet.worker("kita-worker", workflows=[my_task, parse_file_task, ingest_file_task])
    worker.start()

if __name__ == "__main__":
    main()
