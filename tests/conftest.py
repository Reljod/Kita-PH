"""Shared fixtures for the Kita API unit suite.

The whole unit suite is hermetic: no sockets, no Mongo, no Redis, no LLM. Every
external edge is either a `MagicMock` (when we assert on the call) or a real
in-memory fake — `mongomock` / `fakeredis` — when the test needs the collection
to actually behave like a collection.

Environment is pinned here rather than read from `.env.local` on purpose: a unit
run must produce the same result on a laptop, in CI, and in a container with no
network, regardless of which Doppler config happens to be exported.
"""

from __future__ import annotations

import os

# --- Pin the environment BEFORE any `app.*` import ------------------------
# Several modules read os.getenv at import time (auth_service.SECRET_KEY,
# db.Database.connect defaults). Setting these first keeps import-time state
# deterministic instead of inheriting whatever the developer has exported.
_TEST_ENV = {
    "APP_ENV": "test",
    "SECRET_KEY": "unit-test-secret-key-not-a-real-one",
    "MONGO_URI": "mongodb://localhost:27017",
    "MONGO_DB_NAME": "kita_test_db",
    "REDIS_CONNECTION_STRING": "redis://localhost:6379/0",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    "LLM_MODEL": "test/model",
    "SERPER_API_KEY": "test-serper-key",
    "LLAMA_CLOUD_API_KEY": "test-llama-key",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_KEY": "test-supabase-key",
    "SUPABASE_SECRET_KEY": "test-supabase-secret",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "test-password",
    "FACEBOOK_APP_SECRET": "test-fb-secret",
    "FACEBOOK_VERIFY_TOKEN": "test-fb-verify-token",
    "TELEGRAM_WEBHOOK_BASE_URL": "https://kita.test",
    "CORS_ALLOWED_ORIGINS": "http://localhost:3000",
    "HATCHET_CLIENT_TOKEN": "",
    "LOGFIRE_TOKEN": "",
    # Logfire phones home on import unless explicitly silenced.
    "LOGFIRE_SEND_TO_LOGFIRE": "false",
    "LOGFIRE_IGNORE_NO_CONFIG": "1",
}
for _key, _value in _TEST_ENV.items():
    os.environ[_key] = _value

# A deterministic Fernet key so encryption round-trips are reproducible.
from cryptography.fernet import Fernet

os.environ.setdefault("API_KEY_ENCRYPTION_KEY", Fernet.generate_key().decode())

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import mongomock
import pytest
from bson import ObjectId

from app.db import TenantCollection

ORG_ID = "org_test_0001"
USER_ID = "user_test_0001"
AGENT_ID = "agent_test_0001"


# --- Identifiers ----------------------------------------------------------


@pytest.fixture
def org_id() -> str:
    return ORG_ID


@pytest.fixture
def user_id() -> str:
    return USER_ID


@pytest.fixture
def agent_id() -> str:
    return AGENT_ID


@pytest.fixture
def object_id() -> ObjectId:
    """A stable ObjectId, so assertions on ids don't chase a moving target."""
    return ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")


@pytest.fixture
def frozen_now() -> datetime:
    return datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


# --- Mongo ----------------------------------------------------------------


@pytest.fixture
def mongo_client() -> mongomock.MongoClient:
    """A real in-memory Mongo. Use when the test needs query semantics."""
    return mongomock.MongoClient()


@pytest.fixture
def mongo_db(mongo_client: mongomock.MongoClient):
    return mongo_client["kita_test_db"]


@pytest.fixture
def raw_collection(mongo_db):
    """An in-memory collection with real find/insert/update behaviour."""
    return mongo_db["test_collection"]


@pytest.fixture
def tenant_collection(raw_collection) -> TenantCollection:
    """A TenantCollection over a real in-memory collection, scoped to ORG_ID."""
    return TenantCollection(raw_collection, ORG_ID)


@pytest.fixture
def mock_collection() -> MagicMock:
    """A stub collection. Use when the test asserts on *how* Mongo was called.

    Defaults are the empty-result shapes, so a test only sets up the calls it
    actually cares about.
    """
    collection = MagicMock(name="collection")
    collection.find_one.return_value = None
    collection.find.return_value = iter([])
    collection.aggregate.return_value = iter([])
    collection.count_documents.return_value = 0

    insert_result = MagicMock()
    insert_result.inserted_id = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")
    collection.insert_one.return_value = insert_result

    update_result = MagicMock()
    update_result.matched_count = 1
    update_result.modified_count = 1
    collection.update_one.return_value = update_result
    collection.update_many.return_value = update_result

    delete_result = MagicMock()
    delete_result.deleted_count = 1
    collection.delete_one.return_value = delete_result
    collection.delete_many.return_value = delete_result

    return collection


@pytest.fixture
def mock_tenant_collection(mock_collection: MagicMock) -> TenantCollection:
    return TenantCollection(mock_collection, ORG_ID)


@pytest.fixture
def patched_db(monkeypatch, mongo_client):
    """Point the Database at an in-memory Mongo for the duration of a test.

    The class attributes are what matter: the collection accessors are
    classmethods reading `cls.db`, so services that call
    `Database.get_users_collection()` bypass the module-level `db` instance
    entirely. Both are patched so either access style resolves.
    """
    from app.db import Database, db

    monkeypatch.setattr(Database, "client", mongo_client, raising=False)
    monkeypatch.setattr(Database, "db", mongo_client["kita_test_db"], raising=False)
    monkeypatch.setattr(db, "client", mongo_client, raising=False)
    monkeypatch.setattr(db, "db", mongo_client["kita_test_db"], raising=False)
    return db


# --- Redis ----------------------------------------------------------------


@pytest.fixture
async def fake_redis():
    """A real async Redis implementation, in memory."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def mock_redis() -> AsyncMock:
    """A stub async Redis. Use when asserting on cache calls."""
    client = AsyncMock(name="redis")
    client.get.return_value = None
    client.set.return_value = True
    client.incr.return_value = 1
    client.expire.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = 0
    return client


@pytest.fixture(autouse=True)
def _reset_redis_singleton():
    """RedisService caches its client on the class; leaking it across tests
    would let one test's fake serve another's assertions."""
    from app.services.redis_service import RedisService

    RedisService._client = None
    yield
    RedisService._client = None


@pytest.fixture(autouse=True)
def _reset_service_registry():
    """`get_services` memoises a registry per org_id — clear it between tests
    so a registry built against one test's mocks can't serve the next."""
    from app.dependencies import services as services_module

    services_module._registries.clear()
    services_module._event_service = None
    services_module._facebook_service = None
    services_module._org_service = None
    services_module._web_search_service = None
    yield
    services_module._registries.clear()


# --- Service doubles ------------------------------------------------------


@pytest.fixture
def mock_llm_service() -> MagicMock:
    service = MagicMock(name="llm_service")
    service.get_llm.return_value = None
    service.get_llms.return_value = []
    return service


@pytest.fixture
def mock_event_service() -> MagicMock:
    service = MagicMock(name="event_service")
    service.push_event = MagicMock(return_value=None)
    return service


@pytest.fixture
def mock_agent_service() -> MagicMock:
    service = MagicMock(name="agent_service")
    service.get_agent.return_value = None
    service.get_agents.return_value = []
    return service


@pytest.fixture
def mock_web_search_service() -> MagicMock:
    service = MagicMock(name="web_search_service")
    service.search = AsyncMock(return_value=[])
    return service


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- Helpers --------------------------------------------------------------


@pytest.fixture
def make_cursor():
    """Build a stub cursor supporting the chained calls PyMongo returns.

    `collection.find(...).sort(...).skip(...).limit(...)` has to keep returning
    something chainable, and finally iterate over the documents.
    """

    def _make(documents: list[dict[str, Any]]) -> MagicMock:
        cursor = MagicMock(name="cursor")
        cursor.__iter__ = lambda self: iter(documents)
        cursor.sort.return_value = cursor
        cursor.skip.return_value = cursor
        cursor.limit.return_value = cursor
        return cursor

    return _make
