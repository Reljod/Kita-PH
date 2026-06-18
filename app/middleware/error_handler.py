import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions.base import KitaException
from app.exceptions.system import KitaValidationError
from app.utils.logger import ctx_trace_id

logger = logging.getLogger("app.middleware.error_handler")

def setup_error_handlers(app: FastAPI):
    """Registers global exception handlers on the FastAPI app instance."""

    @app.exception_handler(KitaException)
    async def kita_exception_handler(request: Request, exc: KitaException):
        # Determine log level based on status code
        if exc.status_code >= 500:
            logger.error(
                f"KitaException [{exc.code}] (Status {exc.status_code}): {exc.message}",
                extra={"details": exc.details, "code": exc.code},
                exc_info=True
            )
        else:
            logger.warning(
                f"KitaException [{exc.code}] (Status {exc.status_code}): {exc.message}",
                extra={"details": exc.details, "code": exc.code}
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "trace_id": ctx_trace_id.get()
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = {"errors": exc.errors()}
        logger.warning(
            f"Validation error: {exc.errors()}",
            extra={"details": details}
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "SYSTEM_VALIDATION_ERROR",
                    "message": "Input validation failed.",
                    "details": details,
                    "trace_id": ctx_trace_id.get()
                }
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Map standard HTTP exceptions to our standard error code naming conventions
        code = "HTTP_ERROR"
        if exc.status_code == 404:
            code = "RESOURCE_NOT_FOUND"
        elif exc.status_code == 401:
            code = "AUTH_UNAUTHORIZED"
        elif exc.status_code == 403:
            code = "AUTH_FORBIDDEN"
        elif exc.status_code == 400:
            code = "BAD_REQUEST"
        elif exc.status_code == 422:
            code = "SYSTEM_VALIDATION_ERROR"
        elif exc.status_code >= 500:
            code = "SYSTEM_INTERNAL_ERROR"

        logger.warning(
            f"HTTPException [{code}] (Status {exc.status_code}): {exc.detail}"
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": exc.detail,
                    "details": {},
                    "trace_id": ctx_trace_id.get()
                }
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        # Check if the environment is local to return developer-friendly details
        app_env = os.getenv("APP_ENV", "local").lower()
        details = {}
        if app_env == "local":
            details = {"message": str(exc), "type": type(exc).__name__}

        logger.error(
            f"Unhandled exception: {str(exc)}",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "SYSTEM_INTERNAL_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "details": details,
                    "trace_id": ctx_trace_id.get()
                }
            }
        )
