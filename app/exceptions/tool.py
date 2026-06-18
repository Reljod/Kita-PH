from typing import Optional, Dict, Any
from app.exceptions.base import ToolException

class ToolNotFoundError(ToolException):
    code = "TOOL_NOT_FOUND"
    status_code = 404
    
    def __init__(self, tool_name: str, message: Optional[str] = None):
        msg = message or f"Tool '{tool_name}' not found"
        super().__init__(msg, details={"tool_name": tool_name})

class ToolRegistrationError(ToolException):
    code = "TOOL_REGISTRATION_FAILED"
    status_code = 400
    
    def __init__(self, tool_name: str, error_message: str):
        super().__init__(
            f"Failed to register tool '{tool_name}': {error_message}",
            details={"tool_name": tool_name, "error": error_message}
        )

class ToolAgentCreationError(ToolException):
    code = "TOOL_AGENT_CREATION_FAILED"
    status_code = 500

class ToolDelegationError(ToolException):
    code = "TOOL_DELEGATION_FAILED"
    status_code = 500

class ToolFileError(ToolException):
    code = "TOOL_FILE_OPERATION_FAILED"
    status_code = 500

class ToolGraphRagError(ToolException):
    code = "TOOL_GRAPH_RAG_QUERY_FAILED"
    status_code = 500

class ToolLlmError(ToolException):
    code = "TOOL_LLM_COMPLETION_FAILED"
    status_code = 500

class ToolMemoryError(ToolException):
    code = "TOOL_MEMORY_OPERATION_FAILED"
    status_code = 500

class ToolParseError(ToolException):
    code = "TOOL_PARSE_FAILED"
    status_code = 500

class ToolWebSearchError(ToolException):
    code = "TOOL_WEB_SEARCH_FAILED"
    status_code = 500
