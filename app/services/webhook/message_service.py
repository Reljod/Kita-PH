from typing import Any, Dict, Protocol, runtime_checkable

@runtime_checkable
class MessageService(Protocol):
    def verify_webhook(self, mode: str, token: str, challenge: str) -> str:
        """
        Verify the webhook from the platform.
        """
        ...

    async def handle_webhook_event(self, data: Dict[str, Any]) -> None:
        """
        Process incoming webhook events.
        """
        ...
