import logfire
from dotenv import load_dotenv
from app.db import db
from app.workers.hatchet import hatchet
from app.workers.workflows.first_workflow import my_task
from app.workers.workflows.parse_workflow import parse_file_task
from app.workers.workflows.ingest_workflow import ingest_file_task

def main():
    # Load environment variables
    load_dotenv(".env.local")
    load_dotenv()
    
    # Configure Logfire
    logfire.configure()
    logfire.instrument_pydantic_ai()
    
    # Initialize database connection
    db.connect()
    
    worker = hatchet.worker("kita-worker", workflows=[my_task, parse_file_task, ingest_file_task])
    worker.start()

if __name__ == "__main__":
    main()
