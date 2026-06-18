from typing import Optional, Dict, Any
from app.exceptions.base import AuthException

class UnauthorizedError(AuthException):
    code = "AUTH_UNAUTHORIZED"
    status_code = 401

class ForbiddenError(AuthException):
    code = "AUTH_FORBIDDEN"
    status_code = 403

class InvalidApiKeyOrClientError(AuthException):
    code = "AUTH_INVALID_KEY_OR_ID"
    status_code = 401
