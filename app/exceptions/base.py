from typing import Optional, Dict, Any

class KitaException(Exception):
    """Base exception class for Kita API. All enterprise exceptions inherit from this."""
    code: str = "SYSTEM_INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details
            }
        }


class SystemException(KitaException):
    """Parent category for system and configuration errors."""
    code = "SYSTEM_INTERNAL_ERROR"
    status_code = 500


class AuthException(KitaException):
    """Parent category for authentication and authorization errors."""
    code = "AUTH_UNAUTHORIZED"
    status_code = 401


class AgentException(KitaException):
    """Parent category for agent execution and lifecycle errors."""
    code = "AGENT_RUN_FAILED"
    status_code = 500


class ToolException(KitaException):
    """Parent category for tool execution errors."""
    code = "TOOL_EXECUTION_FAILED"
    status_code = 500


class RagException(KitaException):
    """Parent category for RAG and vector database errors."""
    code = "RAG_QUERY_FAILED"
    status_code = 500


class MemoryException(KitaException):
    """Parent category for agent memory-related errors."""
    code = "MEMORY_OPERATION_FAILED"
    status_code = 500


class FileException(KitaException):
    """Parent category for file storage, upload, and parsing errors."""
    code = "FILE_UPLOAD_FAILED"
    status_code = 500
