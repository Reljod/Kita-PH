from app.exceptions.base import KitaException


class IntegrationException(KitaException):
    """Parent category for third-party channel integration errors."""

    code = "INTEGRATION_ERROR"
    status_code = 502


class IntegrationNotConfiguredError(IntegrationException):
    code = "INTEGRATION_NOT_CONFIGURED"
    status_code = 404

    def __init__(self, provider: str, message: str = None):
        super().__init__(
            message or f"No {provider} integration is connected for this organization.",
            {"provider": provider},
        )


class IntegrationCredentialError(IntegrationException):
    """The credential the org supplied was rejected by the provider."""

    code = "INTEGRATION_INVALID_CREDENTIAL"
    status_code = 400


class IntegrationProviderError(IntegrationException):
    """The provider accepted the credential but failed the call."""

    code = "INTEGRATION_PROVIDER_ERROR"
    status_code = 502


class TelegramThreadNotFoundError(IntegrationException):
    code = "INTEGRATION_THREAD_NOT_FOUND"
    status_code = 404

    def __init__(self, thread_id: str):
        super().__init__(
            f"Telegram conversation {thread_id} was not found.",
            {"thread_id": thread_id},
        )
