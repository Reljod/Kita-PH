from typing import Optional, Dict, Any
from app.exceptions.base import MemoryException

class MemoryNotFoundError(MemoryException):
    code = "MEMORY_NOT_FOUND"
    status_code = 404
    
    def __init__(self, memory_id: str, message: Optional[str] = None):
        msg = message or f"Memory record '{memory_id}' not found"
        super().__init__(msg, details={"memory_id": memory_id})

class MemoryOperationFailedError(MemoryException):
    code = "MEMORY_OPERATION_FAILED"
    status_code = 500
