import os
from hatchet_sdk import Hatchet

# Initialize Hatchet client safely or use a mock if token is missing
token = os.getenv("HATCHET_CLIENT_TOKEN")

if token:
    try:
        hatchet = Hatchet()
    except Exception as e:
        print(f"Failed to initialize Hatchet client: {e}. Falling back to mock client.")
        token = None

if not token:
    class MockHatchet:
        def __init__(self):
            class MockEvent:
                async def aio_push(self, *args, **kwargs):
                    return None
            self.event = MockEvent()

        def task(self, *args, **kwargs):
            return lambda func: func

        def workflow(self, *args, **kwargs):
            return lambda func: func

        def worker(self, *args, **kwargs):
            class MockWorker:
                def start(self):
                    print("Hatchet client not configured. Worker not starting.")
                    import time
                    while True:
                        time.sleep(3600)
            return MockWorker()

    hatchet = MockHatchet()

