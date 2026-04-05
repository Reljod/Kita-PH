from app.workers.hatchet import hatchet
from app.workers.workflows.first_workflow import my_task

def main():
    worker = hatchet.worker("test-worker", workflows=[my_task])
    worker.start()

if __name__ == "__main__":
    main()
