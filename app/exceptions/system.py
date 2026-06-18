from typing import Optional, Dict, Any
from app.exceptions.base import SystemException

class KitaDatabaseError(SystemException):
    code = "SYSTEM_DATABASE_ERROR"
    status_code = 500

class KitaRedisError(SystemException):
    code = "SYSTEM_REDIS_ERROR"
    status_code = 500

class KitaValidationError(SystemException):
    code = "SYSTEM_VALIDATION_ERROR"
    status_code = 422

class SystemConfigurationError(SystemException):
    code = "SYSTEM_CONFIG_ERROR"
    status_code = 500
