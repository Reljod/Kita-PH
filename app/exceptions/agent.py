from typing import Optional, Dict, Any
from app.exceptions.base import AgentException

class AgentNotFoundError(AgentException):
    code = "AGENT_NOT_FOUND"
    status_code = 404
    
    def __init__(self, agent_id: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        msg = message or f"Agent '{agent_id}' not found"
        ext_details = {"agent_id": agent_id}
        if details:
            ext_details.update(details)
        super().__init__(msg, details=ext_details)

class AgentVersionNotFoundError(AgentException):
    code = "AGENT_VERSION_NOT_FOUND"
    status_code = 404
    
    def __init__(self, agent_id: str, version: int, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        msg = message or f"Agent '{agent_id}' version {version} not found"
        ext_details = {"agent_id": agent_id, "version": version}
        if details:
            ext_details.update(details)
        super().__init__(msg, details=ext_details)

class AgentRunFailedError(AgentException):
    code = "AGENT_RUN_FAILED"
    status_code = 500
    
    def __init__(self, agent_id: str, error_message: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        msg = message or f"Agent run failed for agent '{agent_id}': {error_message}"
        ext_details = {"agent_id": agent_id, "error": error_message}
        if details:
            ext_details.update(details)
        super().__init__(msg, details=ext_details)

class AgentRunStreamFailedError(AgentException):
    code = "AGENT_RUN_STREAM_FAILED"
    status_code = 500
    
    def __init__(self, agent_id: str, error_message: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        msg = message or f"Agent stream run failed for agent '{agent_id}': {error_message}"
        ext_details = {"agent_id": agent_id, "error": error_message}
        if details:
            ext_details.update(details)
        super().__init__(msg, details=ext_details)

class ChatNotFoundError(AgentException):
    code = "CHAT_NOT_FOUND"
    status_code = 404
    
    def __init__(self, chat_id: str, message: Optional[str] = None):
        msg = message or f"Chat '{chat_id}' not found"
        super().__init__(msg, details={"chat_id": chat_id})
