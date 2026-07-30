from app.exceptions.base import (
    KitaException,
    SystemException,
    AuthException,
    AgentException,
    ToolException,
    RagException,
    MemoryException,
    FileException,
)
from app.exceptions.system import (
    KitaDatabaseError,
    KitaRedisError,
    KitaValidationError,
    SystemConfigurationError,
)
from app.exceptions.auth import (
    UnauthorizedError,
    ForbiddenError,
    InvalidApiKeyOrClientError,
    AuthSessionExpiredError,
)
from app.exceptions.agent import (
    AgentNotFoundError,
    AgentVersionNotFoundError,
    AgentRunFailedError,
    AgentRunStreamFailedError,
    ChatNotFoundError,
)
from app.exceptions.tool import (
    ToolNotFoundError,
    ToolRegistrationError,
    ToolAgentCreationError,
    ToolDelegationError,
    ToolFileError,
    ToolGraphRagError,
    ToolLlmError,
    ToolMemoryError,
    ToolParseError,
    ToolWebSearchError,
)
from app.exceptions.rag import RagQueryFailedError, RagEnrichmentFailedError
from app.exceptions.memory import MemoryNotFoundError, MemoryOperationFailedError
from app.exceptions.file import (
    KitaFileNotFoundError,
    FileUploadFailedError,
    FileParsingFailedError,
)
from app.exceptions.integration import (
    IntegrationException,
    IntegrationNotConfiguredError,
    IntegrationCredentialError,
    IntegrationProviderError,
    TelegramThreadNotFoundError,
)

# This module exists purely to re-export the exception hierarchy, which
# ruff otherwise reads as unused imports. Naming them here states the intent
# and keeps the lint gate meaningful for anyone editing this file.
__all__ = [
    "KitaException",
    "SystemException",
    "AuthException",
    "AgentException",
    "ToolException",
    "RagException",
    "MemoryException",
    "FileException",
    "KitaDatabaseError",
    "KitaRedisError",
    "KitaValidationError",
    "SystemConfigurationError",
    "UnauthorizedError",
    "ForbiddenError",
    "InvalidApiKeyOrClientError",
    "AuthSessionExpiredError",
    "AgentNotFoundError",
    "AgentVersionNotFoundError",
    "AgentRunFailedError",
    "AgentRunStreamFailedError",
    "ChatNotFoundError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolAgentCreationError",
    "ToolDelegationError",
    "ToolFileError",
    "ToolGraphRagError",
    "ToolLlmError",
    "ToolMemoryError",
    "ToolParseError",
    "ToolWebSearchError",
    "RagQueryFailedError",
    "RagEnrichmentFailedError",
    "MemoryNotFoundError",
    "MemoryOperationFailedError",
    "KitaFileNotFoundError",
    "FileUploadFailedError",
    "FileParsingFailedError",
    "IntegrationException",
    "IntegrationNotConfiguredError",
    "IntegrationCredentialError",
    "IntegrationProviderError",
    "TelegramThreadNotFoundError",
]
