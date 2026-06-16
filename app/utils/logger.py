import contextvars
import logging
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple
from fastapi import Header

# Define ContextVars for storing request-scoped tracing and authentication context
ctx_org_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("org_id", default=None)
ctx_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("user_id", default=None)
ctx_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
ctx_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
ctx_client_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("client_id", default=None)

def set_logging_context(
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    client_id: Optional[str] = None,
):
    """Sets the provided request context values in contextvars and OpenTelemetry span attributes."""
    if org_id is not None:
        ctx_org_id.set(org_id)
    if user_id is not None:
        ctx_user_id.set(user_id)
    if request_id is not None:
        ctx_request_id.set(request_id)
    if trace_id is not None:
        ctx_trace_id.set(trace_id)
    if client_id is not None:
        ctx_client_id.set(client_id)

    from opentelemetry import trace
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        if org_id is not None:
            current_span.set_attribute("org_id", org_id)
        if user_id is not None:
            current_span.set_attribute("user_id", user_id)
        if request_id is not None:
            current_span.set_attribute("request_id", request_id)
        if trace_id is not None:
            current_span.set_attribute("trace_id", trace_id)
        if client_id is not None:
            current_span.set_attribute("client_id", client_id)

def clear_logging_context():
    """Clears all stored context variables."""
    ctx_org_id.set(None)
    ctx_user_id.set(None)
    ctx_request_id.set(None)
    ctx_trace_id.set(None)
    ctx_client_id.set(None)

class ContextFilter(logging.Filter):
    """
    Filter that injects request-scoped attributes (org_id, user_id, request_id, trace_id, client_id)
    directly into the LogRecord.
    This ensures that downstream handlers like Logfire can capture these as structured attributes
    and include them in their telemetry.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        org_id = ctx_org_id.get()
        user_id = ctx_user_id.get()
        request_id = ctx_request_id.get()
        trace_id = ctx_trace_id.get()
        client_id = ctx_client_id.get()

        # Inject as attributes on the record object (which Logfire picks up)
        if org_id is not None:
            record.org_id = org_id
        if user_id is not None:
            record.user_id = user_id
        if request_id is not None:
            record.request_id = request_id
        if trace_id is not None:
            record.trace_id = trace_id
        if client_id is not None:
            record.client_id = client_id

        return True

class LogFormatter(logging.Formatter):
    """
    Custom formatter that formats LogRecords.
    Supports structured JSON layout (production style) or clean text console layout.
    """
    def __init__(self, use_json: bool = False):
        super().__init__()
        self.use_json = use_json

    def format(self, record: logging.LogRecord) -> str:
        org_id = getattr(record, "org_id", "-")
        user_id = getattr(record, "user_id", "-")
        request_id = getattr(record, "request_id", "-")
        trace_id = getattr(record, "trace_id", "-")
        client_id = getattr(record, "client_id", "-")

        if self.use_json:
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "org_id": None if org_id == "-" else org_id,
                "user_id": None if user_id == "-" else user_id,
                "request_id": None if request_id == "-" else request_id,
                "trace_id": None if trace_id == "-" else trace_id,
                "client_id": None if client_id == "-" else client_id,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                log_data["error"] = self.formatException(record.exc_info)
            elif hasattr(record, "error") and getattr(record, "error"):
                log_data["error"] = str(record.error)
            else:
                log_data["error"] = None
                
            return json.dumps(log_data)
        else:
            timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
            exc_str = ""
            if record.exc_info:
                exc_str = f"\n{self.formatException(record.exc_info)}"
            return f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}{exc_str}"

def setup_logging():
    """Configures root logging with custom LogFormatter, handling Logfire integration if configured."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    if log_level_str not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level_str = "INFO"
    level = getattr(logging, log_level_str)
    
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    use_json = log_format == "json"
    
    # Instantiate the ContextFilter
    context_filter = ContextFilter()

    # Configure Console Stream Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter(use_json=use_json))
    console_handler.setLevel(level)
    console_handler.addFilter(context_filter)
    
    handlers = [console_handler]
    
    # Check if Logfire Logging Handler is available
    try:
        import logfire
        logfire_handler = logfire.LogfireLoggingHandler()
        logfire_handler.addFilter(context_filter)
        handlers.append(logfire_handler)
    except ImportError:
        pass
        
    logging.basicConfig(
        handlers=handlers,
        level=level,
        force=True
    )
    
    # Ensure uvicorn and other server loggers propagate to root handler to get formatting
    for logger_name in ("uvicorn", "uvicorn.error", "fastapi"):
        l = logging.getLogger(logger_name)
        l.handlers = []
        l.propagate = True

    # Disable uvicorn access logging to prevent duplicate requests logs, since logfire handles request tracing
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers = []
    uvicorn_access.propagate = False
    uvicorn_access.setLevel(logging.WARNING)

class CorrelationIdMiddleware:
    """
    Pure ASGI middleware that manages the request-response correlation ID lifecycle.
    Unlike Starlette's BaseHTTPMiddleware, this preserves Python contextvars perfectly
    across the entire request task context.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Parse headers from ASGI scope (names are lowercase bytes)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        
        req_id_bytes = headers.get(b"x-request-id", b"")
        req_id = req_id_bytes.decode("latin1") if req_id_bytes else str(uuid.uuid4())
        
        tr_id_bytes = headers.get(b"x-trace-id", b"")
        tr_id = tr_id_bytes.decode("latin1") if tr_id_bytes else str(uuid.uuid4())
        
        client_id_bytes = headers.get(b"x-client-id", b"")
        client_id = client_id_bytes.decode("latin1") if client_id_bytes else None

        # Set task-local / context-local variables
        token_req = ctx_request_id.set(req_id)
        token_trace = ctx_trace_id.set(tr_id)
        token_client = ctx_client_id.set(client_id)
        token_user = ctx_user_id.set(None)
        token_org = ctx_org_id.set(None)

        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("request_id", req_id)
            current_span.set_attribute("trace_id", tr_id)
            if client_id:
                current_span.set_attribute("client_id", client_id)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_headers = message.setdefault("headers", [])
                response_headers.append((b"x-request-id", req_id.encode("latin1")))
                response_headers.append((b"x-trace-id", tr_id.encode("latin1")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ctx_request_id.reset(token_req)
            ctx_trace_id.reset(token_trace)
            ctx_client_id.reset(token_client)
            ctx_user_id.reset(token_user)
            ctx_org_id.reset(token_org)

async def get_global_headers(
    x_request_id: Optional[str] = Header(None, alias="x-request-id"),
    x_trace_id: Optional[str] = Header(None, alias="x-trace-id"),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    x_client_id: Optional[str] = Header(None, alias="x-client-id"),
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    FastAPI dependency that exposes request tracing and client credential headers.
    Synchronizes incoming header values to contextvars and OpenTelemetry span attributes.
    """
    set_logging_context(
        request_id=x_request_id,
        trace_id=x_trace_id,
        client_id=x_client_id
    )
    return x_request_id, x_trace_id, x_api_key, x_client_id
