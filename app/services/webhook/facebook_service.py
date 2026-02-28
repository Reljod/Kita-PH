import os
import hmac
import hashlib
from typing import Any, Dict
from fastapi import HTTPException

from app.services.organization_service import OrganizationService
from app.services.chat_service import ChatService
from app.services.agent_service import AgentService
from app.services.llm_service import LlmService
from app.services.agents.creator_agent import CreatorAgentService
from app.db import db, TenantCollection
from app.models.chat import ChatCreateRequest

class FacebookService:
    def __init__(self):
        self.verify_token = os.getenv("FACEBOOK_VERIFY_TOKEN", "my_secret_verify_token")
        self.app_secret = os.getenv("FACEBOOK_APP_SECRET", "")

    def verify_signature(self, payload: bytes, signature_header: str) -> bool:
        if not signature_header or not self.app_secret:
            return False
            
        parts = signature_header.split("=")
        if len(parts) != 2 or parts[0] != "sha256":
            return False
            
        expected_signature = hmac.new(
            self.app_secret.encode("utf-8"),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, parts[1])

    def verify_webhook(self, mode: str, token: str, challenge: str) -> str:
        """
        Verifies the challenge token from Facebook.
        """
        if mode == "subscribe" and token == self.verify_token:
            return challenge
        else:
            raise HTTPException(status_code=403, detail="Verification failed")

    async def handle_webhook_event(self, data: Dict[str, Any]) -> None:
        """
        Processes events from Facebook Messenger.
        """
        if data.get("object") == "page":
            for entry in data.get("entry", []):
                # Iterate over messaging events
                for messaging_event in entry.get("messaging", []):
                    if messaging_event.get("message"):
                        await self._handle_message(messaging_event)
                    elif messaging_event.get("postback"):
                        await self._handle_postback(messaging_event)
        else:
            # Not a page event
            raise HTTPException(status_code=404)

    async def _handle_message(self, event: Dict[str, Any]) -> None:
        print("Event: ", event)
        sender_id = event.get("sender", {}).get("id")
        recipient_id = event.get("recipient", {}).get("id")
        message = event.get("message")

        if not message:
            return
        
        message_text = None
        if isinstance(message, dict):
            message_text = message.get("text")
        else:
            message_text = message

        if not message_text:
            return

        print(f"Received message from facebook user {sender_id} to page {recipient_id}: {message_text}")
        
        org_service = OrganizationService()
        org = org_service.get_org_by_integration_id("facebook_page_id", recipient_id)
        if not org:
            print(f"No organization mapping found for Facebook Page ID: {sender_id}")
            return
            
        org_id = org.id
        print(f"Mapped facebook page {sender_id} to org_id: {org_id}")
        
        llm_service_coll = TenantCollection(db.get_llms_collection(), org_id)
        llm_service = LlmService(llm_service_coll)
        prompt_writer = CreatorAgentService(org_id)
        agent_coll = TenantCollection(db.get_agents_collection(), org_id)
        agent_service = AgentService(llm_service=llm_service, prompt_writer_service=prompt_writer, collection=agent_coll)

        chat_coll = TenantCollection(db.get_chats_collection(), org_id)
        chat_service = ChatService(agent_service, chat_coll)
        
        req = ChatCreateRequest(message=message_text)
        try:
            print(f"Message text: {message_text}")
            # res = await chat_service.create_chat(req)
            print(f"Successfully processed webhook message and created chat for org {org_id}.")
            # Further logic to send the response back to facebook can be added here
        except Exception as e:
            print(f"Error handling facebook webhook message for org {org_id}: {e}")

    async def _handle_postback(self, event: Dict[str, Any]) -> None:
        sender_id = event.get("sender", {}).get("id")
        recipient_id = event.get("recipient", {}).get("id")
        payload = event.get("postback", {}).get("payload")
        print(f"Received postback from {sender_id}: {payload}")
        # TODO: Implement postback handling logic
