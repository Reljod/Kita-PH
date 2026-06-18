import unittest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from app.exceptions.base import KitaException
from app.exceptions.agent import AgentNotFoundError
from app.exceptions.tool import ToolWebSearchError
from app.exceptions.system import KitaValidationError
from app.middleware.error_handler import setup_error_handlers
from app.utils.logger import ctx_trace_id

class TestErrorHandlers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Create a test FastAPI application
        cls.test_app = FastAPI()

        # Setup dummy routes that raise various exceptions
        @cls.test_app.get("/test-kita-error")
        def raise_kita_error():
            raise AgentNotFoundError("agent_123")

        @cls.test_app.get("/test-nested-kita-error")
        def raise_nested_kita_error():
            raise ToolWebSearchError("web_search failed", details={"query": "test query"})

        @cls.test_app.get("/test-auth-session-expired")
        def raise_auth_session_expired():
            from app.exceptions.auth import AuthSessionExpiredError
            raise AuthSessionExpiredError("Your session has expired")

        @cls.test_app.get("/test-std-http-error")
        def raise_std_http_error():
            raise HTTPException(status_code=403, detail="Access denied")

        class DummyItem(BaseModel):
            name: str
            price: float

        @cls.test_app.post("/test-post-validation")
        def post_validation(item: DummyItem):
            return {"message": "success"}

        @cls.test_app.get("/test-unhandled-error")
        def raise_unhandled_error():
            raise ZeroDivisionError("division by zero")

        # Initialize exception handlers
        setup_error_handlers(cls.test_app)
        cls.client = TestClient(cls.test_app, raise_server_exceptions=False)

    def test_kita_exception_response(self):
        # Set a trace ID in context
        token = ctx_trace_id.set("test-correlation-id-123")
        try:
            response = self.client.get("/test-kita-error")
            self.assertEqual(response.status_code, 404)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "AGENT_NOT_FOUND")
            self.assertIn("agent_123", data["error"]["message"])
            self.assertEqual(data["error"]["details"], {"agent_id": "agent_123"})
            self.assertEqual(data["error"]["trace_id"], "test-correlation-id-123")
        finally:
            ctx_trace_id.reset(token)

    def test_nested_kita_exception_response(self):
        token = ctx_trace_id.set("test-correlation-id-456")
        try:
            response = self.client.get("/test-nested-kita-error")
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "TOOL_WEB_SEARCH_FAILED")
            self.assertIn("web_search failed", data["error"]["message"])
            self.assertEqual(data["error"]["details"], {"query": "test query"})
            self.assertEqual(data["error"]["trace_id"], "test-correlation-id-456")
        finally:
            ctx_trace_id.reset(token)

    def test_auth_session_expired_response(self):
        token = ctx_trace_id.set("test-expired-session-id")
        try:
            response = self.client.get("/test-auth-session-expired")
            self.assertEqual(response.status_code, 401)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "AUTH_SESSION_EXPIRED")
            self.assertEqual(data["error"]["message"], "Your session has expired")
            self.assertEqual(data["error"]["trace_id"], "test-expired-session-id")
        finally:
            ctx_trace_id.reset(token)

    def test_starlette_http_exception_response(self):
        token = ctx_trace_id.set("test-correlation-id-789")
        try:
            response = self.client.get("/test-std-http-error")
            self.assertEqual(response.status_code, 403)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "AUTH_FORBIDDEN")
            self.assertEqual(data["error"]["message"], "Access denied")
            self.assertEqual(data["error"]["trace_id"], "test-correlation-id-789")
        finally:
            ctx_trace_id.reset(token)

    def test_pydantic_validation_exception_response(self):
        token = ctx_trace_id.set("test-correlation-id-999")
        try:
            # Pass invalid body
            response = self.client.post("/test-post-validation", json={"price": "not-a-float"})
            self.assertEqual(response.status_code, 422)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "SYSTEM_VALIDATION_ERROR")
            self.assertEqual(data["error"]["message"], "Input validation failed.")
            self.assertIn("errors", data["error"]["details"])
            self.assertGreater(len(data["error"]["details"]["errors"]), 0)
            self.assertEqual(data["error"]["trace_id"], "test-correlation-id-999")
        finally:
            ctx_trace_id.reset(token)

    def test_unhandled_exception_response(self):
        token = ctx_trace_id.set("test-correlation-id-000")
        try:
            response = self.client.get("/test-unhandled-error")
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "SYSTEM_INTERNAL_ERROR")
            self.assertIn("unexpected internal server error", data["error"]["message"].lower())
            self.assertEqual(data["error"]["trace_id"], "test-correlation-id-000")
        finally:
            ctx_trace_id.reset(token)
