from typing import Dict, Any, Protocol, Optional
import os
from hatchet_sdk import Hatchet

class IEventService(Protocol):
    async def push(self, event_key: str, payload: Dict[str, Any]):
        ...

class HatchetEventService(IEventService):
    def __init__(self):
        token = os.getenv("HATCHET_CLIENT_TOKEN")
        if not token:
            print("HATCHET_CLIENT_TOKEN not set. Events will not be pushed.")
            self.client = None
        else:
            self.client = Hatchet()

    async def push(self, event_key: str, payload: Dict[str, Any]):
        if not self.client:
            print(f"Skipping event {event_key} (Hatchet client not initialized)")
            return
            
        try:
            await self.client.event.aio_push(event_key, payload)
        except Exception as e:
            print(f"Error pushing event to Hatchet: {e}")

