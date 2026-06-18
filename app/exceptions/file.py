from typing import Optional, Dict, Any
from app.exceptions.base import FileException

class KitaFileNotFoundError(FileException):
    code = "FILE_NOT_FOUND"
    status_code = 404
    
    def __init__(self, file_id: str, message: Optional[str] = None):
        msg = message or f"File '{file_id}' not found"
        super().__init__(msg, details={"file_id": file_id})

class FileUploadFailedError(FileException):
    code = "FILE_UPLOAD_FAILED"
    status_code = 500

class FileParsingFailedError(FileException):
    code = "FILE_PARSING_FAILED"
    status_code = 422
