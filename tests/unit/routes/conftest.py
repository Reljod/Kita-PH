"""Helpers for exercising routers in isolation.

Each router is mounted on a bare FastAPI app with its dependencies overridden,
so a route test asserts the handler's own logic — status codes, error shapes,
what it passes to the services — without dragging in Mongo, Redis, or the
API-key middleware that `main.py` wraps around everything.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.error_handler import setup_error_handlers


@pytest.fixture
def make_app():
    """Mount a router on a bare app, with the project's error handlers.

    The handlers matter: without them a KitaException escapes as a 500 and
    every "does this return 404" assertion silently tests the wrong thing.
    """

    def _make(router, overrides: dict | None = None) -> FastAPI:
        app = FastAPI()
        setup_error_handlers(app)
        app.include_router(router)
        for dependency, replacement in (overrides or {}).items():
            app.dependency_overrides[dependency] = replacement
        return app

    return _make


@pytest.fixture
def make_client(make_app):
    def _make(router, overrides: dict | None = None) -> TestClient:
        # raise_server_exceptions=False so an unhandled error surfaces as the
        # 500 a real client would see, rather than re-raising into the test.
        return TestClient(make_app(router, overrides), raise_server_exceptions=False)

    return _make


def override(value):
    """Build a dependency override that yields a fixed object."""

    def _dependency():
        return value

    return _dependency


@pytest.fixture
def auth_service() -> MagicMock:
    service = MagicMock(name="auth_service")
    token = MagicMock()
    token.access_token = "access-token"
    token.refresh_token = "refresh-token"
    token.token_type = "bearer"
    service.generate_tokens.return_value = token
    service.verify_token.return_value = None
    return service


@pytest.fixture
def user_service() -> MagicMock:
    service = MagicMock(name="user_service")
    service.get_user_by_email.return_value = None
    service.get_user_by_id.return_value = None
    return service


@pytest.fixture
def org_service() -> MagicMock:
    service = MagicMock(name="org_service")
    service.get_user_orgs.return_value = []
    service.get_org.return_value = None
    service.get_org_by_code.return_value = None
    return service
