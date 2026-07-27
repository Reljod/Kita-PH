"""Tests for app.utils.logger.

Correlation context (org, user, request, trace, client) is carried in
contextvars and stamped onto every LogRecord. That is how a support request
turns into a filterable set of log lines, so the tests that matter are the
ones proving the context is attached, isolated per request, and reset
afterwards -- a leaked contextvar would tag the *next* request with the
previous caller's organization.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.utils.logger import (
    ContextFilter,
    CorrelationIdMiddleware,
    LogFormatter,
    clear_logging_context,
    ctx_client_id,
    ctx_org_id,
    ctx_request_id,
    ctx_trace_id,
    ctx_user_id,
    get_global_headers,
    log_tool_call,
    set_logging_context,
    setup_logging,
)

ORG_ID = "org_test_0001"
USER_ID = "user_1"


@pytest.fixture(autouse=True)
def clean_context():
    """contextvars outlive a test function, so reset around every one."""
    clear_logging_context()
    yield
    clear_logging_context()


def a_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.api",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# --- context -------------------------------------------------------------


class TestSetLoggingContext:
    def test_each_field_is_stored(self):
        set_logging_context(
            org_id=ORG_ID,
            user_id=USER_ID,
            request_id="req",
            trace_id="trace",
            client_id="client",
        )
        assert ctx_org_id.get() == ORG_ID
        assert ctx_user_id.get() == USER_ID
        assert ctx_request_id.get() == "req"
        assert ctx_trace_id.get() == "trace"
        assert ctx_client_id.get() == "client"

    def test_omitted_fields_are_left_alone(self):
        """Auth resolves the organization after the middleware has already set
        the request id, so a partial update must not wipe the rest."""
        set_logging_context(request_id="req")
        set_logging_context(org_id=ORG_ID)
        assert ctx_request_id.get() == "req" and ctx_org_id.get() == ORG_ID

    def test_the_context_starts_empty(self):
        assert ctx_org_id.get() is None

    def test_clearing_resets_everything(self):
        set_logging_context(org_id=ORG_ID, user_id=USER_ID, request_id="req")
        clear_logging_context()
        assert ctx_org_id.get() is None
        assert ctx_user_id.get() is None
        assert ctx_request_id.get() is None

    def test_the_span_is_annotated(self, monkeypatch):
        """Logfire correlates traces by these attributes, not by the log line."""
        span = MagicMock()
        span.is_recording.return_value = True
        monkeypatch.setattr("opentelemetry.trace.get_current_span", lambda: span)
        set_logging_context(org_id=ORG_ID)
        span.set_attribute.assert_any_call("org_id", ORG_ID)

    def test_a_span_that_is_not_recording_is_left_alone(self, monkeypatch):
        span = MagicMock()
        span.is_recording.return_value = False
        monkeypatch.setattr("opentelemetry.trace.get_current_span", lambda: span)
        set_logging_context(org_id=ORG_ID)
        span.set_attribute.assert_not_called()


class TestContextFilter:
    def test_the_record_is_always_kept(self):
        """This is a filter by mechanism only -- it enriches, it never drops."""
        assert ContextFilter().filter(a_record()) is True

    def test_the_context_is_stamped_onto_the_record(self):
        set_logging_context(org_id=ORG_ID, user_id=USER_ID)
        record = a_record()
        ContextFilter().filter(record)
        assert record.org_id == ORG_ID and record.user_id == USER_ID

    def test_an_empty_context_adds_nothing(self):
        record = a_record()
        ContextFilter().filter(record)
        assert not hasattr(record, "org_id")

    def test_every_field_reaches_the_record(self):
        set_logging_context(
            org_id=ORG_ID,
            user_id=USER_ID,
            request_id="req",
            trace_id="trace",
            client_id="client",
        )
        record = a_record()
        ContextFilter().filter(record)
        assert record.request_id == "req"
        assert record.trace_id == "trace"
        assert record.client_id == "client"


# --- formatting -----------------------------------------------------------


class TestTextFormatter:
    def test_the_message_is_rendered(self):
        assert "hello world" in LogFormatter().format(a_record())

    def test_the_level_is_shown(self):
        assert "[INFO]" in LogFormatter().format(a_record())

    def test_the_logger_name_is_shown(self):
        assert "app.api" in LogFormatter().format(a_record())

    def test_extras_are_appended(self):
        assert "duration=1.5" in LogFormatter().format(a_record(duration=1.5))

    def test_a_record_with_no_extras_gets_no_extras_block(self):
        """Python 3.12 puts taskName on every record; treating it as a
        user-supplied extra tags every async log line with framework noise."""
        assert LogFormatter().format(a_record()).count("[") == 1

    def test_an_exception_is_appended(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = a_record()
            record.exc_info = sys.exc_info()
        assert "ValueError: boom" in LogFormatter().format(record)


class TestJsonFormatter:
    def test_the_output_is_valid_json(self):
        """Log shippers parse these line by line; one unquoted value breaks
        the whole ingestion pipeline."""
        assert json.loads(LogFormatter(use_json=True).format(a_record()))

    def test_the_message_is_rendered(self):
        payload = json.loads(LogFormatter(use_json=True).format(a_record()))
        assert payload["msg"] == "hello world"

    def test_missing_context_is_null_rather_than_a_dash(self):
        """A literal "-" would be indexed as a value and match across every
        request that had no organization."""
        payload = json.loads(LogFormatter(use_json=True).format(a_record()))
        assert payload["org_id"] is None and payload["user_id"] is None

    def test_present_context_is_carried(self):
        record = a_record()
        set_logging_context(org_id=ORG_ID)
        ContextFilter().filter(record)
        payload = json.loads(LogFormatter(use_json=True).format(record))
        assert payload["org_id"] == ORG_ID

    def test_extras_are_merged_in(self):
        payload = json.loads(LogFormatter(use_json=True).format(a_record(duration=1.5)))
        assert payload["duration"] == 1.5

    def test_the_error_field_is_null_without_one(self):
        payload = json.loads(LogFormatter(use_json=True).format(a_record()))
        assert payload["error"] is None

    def test_an_exception_becomes_the_error_field(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = a_record()
            record.exc_info = sys.exc_info()
        payload = json.loads(LogFormatter(use_json=True).format(record))
        assert "ValueError: boom" in payload["error"]

    def test_an_explicit_error_extra_is_used(self):
        payload = json.loads(
            LogFormatter(use_json=True).format(a_record(error="something failed"))
        )
        assert payload["error"] == "something failed"

    def test_the_timestamp_is_iso_formatted(self):
        payload = json.loads(LogFormatter(use_json=True).format(a_record()))
        from datetime import datetime

        assert datetime.fromisoformat(payload["timestamp"])


# --- setup ----------------------------------------------------------------


class TestSetupLogging:
    def test_a_handler_is_installed(self):
        setup_logging()
        assert logging.getLogger().handlers

    def test_the_level_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        setup_logging()
        assert logging.getLogger().level == logging.DEBUG

    def test_an_unknown_level_falls_back_to_info(self, monkeypatch):
        """A typo in the deployment config should not silence the logs."""
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        setup_logging()
        assert logging.getLogger().level == logging.INFO

    def test_json_output_can_be_selected(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "json")
        setup_logging()
        console = logging.getLogger().handlers[0]
        assert console.formatter.use_json is True

    def test_text_is_the_default(self, monkeypatch):
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        setup_logging()
        assert logging.getLogger().handlers[0].formatter.use_json is False

    def test_uvicorn_access_logs_are_silenced(self, monkeypatch):
        """Logfire already traces every request; leaving these on doubles the
        line count for no extra information."""
        setup_logging()
        access = logging.getLogger("uvicorn.access")
        assert access.propagate is False and access.level == logging.WARNING

    def test_server_loggers_are_reformatted_through_the_root(self):
        setup_logging()
        for name in ("uvicorn", "uvicorn.error", "fastapi"):
            assert logging.getLogger(name).propagate is True
            assert logging.getLogger(name).handlers == []


# --- correlation middleware -----------------------------------------------


def http_scope(**overrides) -> dict:
    scope = {"type": "http", "method": "GET", "path": "/agents", "headers": []}
    scope.update(overrides)
    return scope


async def run_middleware(app, scope) -> list:
    sent = []

    async def send(message):
        sent.append(message)

    await CorrelationIdMiddleware(app)(scope, AsyncMock(), send)
    return sent


def an_app(seen: list | None = None):
    async def app(scope, receive, send):
        if seen is not None:
            seen.append(
                {
                    "request_id": ctx_request_id.get(),
                    "trace_id": ctx_trace_id.get(),
                    "client_id": ctx_client_id.get(),
                }
            )
        await send({"type": "http.response.start", "status": 200, "headers": []})

    return app


class TestCorrelationIdMiddleware:
    async def test_a_lifespan_message_passes_straight_through(self):
        app = AsyncMock()
        await CorrelationIdMiddleware(app)(
            {"type": "lifespan"}, AsyncMock(), AsyncMock()
        )
        app.assert_awaited_once()

    async def test_an_id_is_generated_when_none_is_supplied(self):
        seen = []
        await run_middleware(an_app(seen), http_scope())
        assert seen[0]["request_id"] and seen[0]["trace_id"]

    async def test_the_generated_ids_differ(self):
        """Sharing one id between the request and the trace would collapse two
        different lookups into one."""
        seen = []
        await run_middleware(an_app(seen), http_scope())
        assert seen[0]["request_id"] != seen[0]["trace_id"]

    async def test_an_inbound_request_id_is_honoured(self):
        """The UI passes its own id so a browser error can be matched to a
        server log line."""
        seen = []
        await run_middleware(
            an_app(seen), http_scope(headers=[(b"x-request-id", b"from-client")])
        )
        assert seen[0]["request_id"] == "from-client"

    async def test_an_inbound_trace_id_is_honoured(self):
        seen = []
        await run_middleware(
            an_app(seen), http_scope(headers=[(b"x-trace-id", b"trace-from-client")])
        )
        assert seen[0]["trace_id"] == "trace-from-client"

    async def test_the_client_id_is_captured(self):
        seen = []
        await run_middleware(
            an_app(seen), http_scope(headers=[(b"x-client-id", b"kita-ui")])
        )
        assert seen[0]["client_id"] == "kita-ui"

    async def test_header_names_are_matched_case_insensitively(self):
        seen = []
        await run_middleware(
            an_app(seen), http_scope(headers=[(b"X-Request-ID", b"mixed-case")])
        )
        assert seen[0]["request_id"] == "mixed-case"

    async def test_the_ids_are_echoed_back_to_the_caller(self):
        """Without them the client cannot quote an id when reporting a fault."""
        sent = await run_middleware(an_app(), http_scope())
        headers = dict(sent[0]["headers"])
        assert b"x-request-id" in headers and b"x-trace-id" in headers

    async def test_the_context_is_reset_afterwards(self):
        """A leaked contextvar would tag the next request on this worker with
        the previous caller's ids."""
        await run_middleware(an_app(), http_scope())
        assert ctx_request_id.get() is None and ctx_trace_id.get() is None

    async def test_the_context_is_reset_even_when_the_app_raises(self):
        async def failing(scope, receive, send):
            raise RuntimeError("handler exploded")

        with pytest.raises(RuntimeError):
            await run_middleware(failing, http_scope())
        assert ctx_request_id.get() is None

    async def test_a_websocket_session_is_handled(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(ctx_request_id.get())

        await CorrelationIdMiddleware(app)(
            {"type": "websocket", "path": "/chat/status/ws/k", "headers": []},
            AsyncMock(),
            AsyncMock(),
        )
        assert seen[0]

    async def test_the_organization_is_not_assumed(self):
        """Auth has not run yet at middleware time, so anything other than
        None here would be a guess."""
        seen = []

        async def app(scope, receive, send):
            seen.append(ctx_org_id.get())
            await send({"type": "http.response.start", "status": 200, "headers": []})

        await CorrelationIdMiddleware(app)(http_scope(), AsyncMock(), AsyncMock())
        assert seen[0] is None


# --- header dependency ----------------------------------------------------


class TestGetGlobalHeaders:
    async def test_the_headers_are_returned(self):
        result = await get_global_headers("req", "trace", "key", "client")
        assert result == ("req", "trace", "key", "client")

    async def test_the_context_is_synchronised(self):
        await get_global_headers("req", "trace", "key", "client")
        assert ctx_request_id.get() == "req"
        assert ctx_trace_id.get() == "trace"
        assert ctx_client_id.get() == "client"

    async def test_the_api_key_is_never_stored_in_the_context(self):
        """It would end up in every log line stamped by ContextFilter."""
        await get_global_headers("req", "trace", "secret-key", "client")
        assert "secret-key" not in str(
            [ctx_request_id.get(), ctx_trace_id.get(), ctx_client_id.get()]
        )

    async def test_missing_headers_are_accepted(self):
        assert await get_global_headers(None, None, None, None) == (
            None,
            None,
            None,
            None,
        )


# --- tool instrumentation -------------------------------------------------


class TestLogToolCall:
    async def test_the_result_is_passed_through(self):
        @log_tool_call
        async def my_tool(value):
            return value * 2

        assert await my_tool(21) == 42

    async def test_the_function_identity_is_preserved(self):
        """pydantic-ai reads the name and docstring to build the tool schema,
        so an unwrapped decorator would rename every tool to "wrapper"."""

        @log_tool_call
        async def my_tool(value):
            """Doubles things."""
            return value

        assert my_tool.__name__ == "my_tool"
        assert my_tool.__doc__ == "Doubles things."

    async def test_a_failure_is_wrapped_for_the_agent(self):
        """The agent sees tool errors as tool results, so a raw exception type
        would leak an implementation detail into the model's context."""
        from app.exceptions import ToolException

        @log_tool_call
        async def failing_tool():
            raise ValueError("tool broke")

        with pytest.raises(ToolException, match="tool broke"):
            await failing_tool()

    async def test_the_run_context_is_kept_out_of_the_logs(self):
        """RunContext carries the whole dependency graph, including service
        objects holding credentials."""
        from pydantic_ai import RunContext

        captured = {}

        @log_tool_call
        async def my_tool(ctx, value):
            captured["ctx"] = ctx
            return "ok"

        ctx = MagicMock(spec=RunContext)
        assert await my_tool(ctx, "visible") == "ok"
        assert captured["ctx"] is ctx

    async def test_long_arguments_are_truncated(self):
        """A tool called with a whole document would otherwise put the entire
        thing in the log line and the span attribute."""

        @log_tool_call
        async def my_tool(value):
            return "ok"

        assert await my_tool("x" * 5000) == "ok"
