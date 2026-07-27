"""Tests for app.routes.chat, app.routes.rag and app.routes.event.

The chat routes are the org-wide twin of the agent-scoped chat endpoints —
same SSE contract, with the agent filter arriving as an `x-agent-id` header.
The event route is the only place that validates a payload against the event
key before handing it to Hatchet, so the mismatch cases matter.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.dependencies import get_chat_service, get_event_service, get_retrieval_service
from app.models.chat import ChatResponse
from app.models.rag import RagResponse
from app.routes import chat as chat_routes
from app.routes import event as event_routes
from app.routes import rag as rag_routes
from app.security import require_org_membership

from .conftest import override

CHAT_ID = "chat_1"
AGENT_ID = "agent_1"
ORG_ID = "org_abc"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def a_chat(**overrides) -> ChatResponse:
    payload = {
        "id": CHAT_ID,
        "messages": [],
        "agent_id": AGENT_ID,
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return ChatResponse(**payload)


def a_rag() -> RagResponse:
    return RagResponse(
        id="rag_1",
        title="Note",
        content="body",
        status="completed",
        created_at=NOW,
        updated_at=NOW,
    )


# --- chat -----------------------------------------------------------------


@pytest.fixture
def chat_service() -> MagicMock:
    service = MagicMock(name="chat_service")
    service.get_chat.return_value = a_chat()
    service.get_all_chats.return_value = [a_chat()]

    async def stream(*args, **kwargs):
        yield {"type": "delta", "text": "hi"}

    service.create_chat_stream = stream
    service.continue_chat_stream = stream
    return service


@pytest.fixture
def chat_client(make_client, chat_service):
    return make_client(
        chat_routes.router,
        {
            get_chat_service: override(chat_service),
            require_org_membership: override(ORG_ID),
        },
    )


class TestChatStreaming:
    def test_a_new_chat_streams(self, chat_client):
        response = chat_client.post("/chat", json={"message": "hi"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_events_are_server_sent_json(self, chat_client):
        response = chat_client.post("/chat", json={"message": "hi"})
        assert json.loads(response.text.split("data: ", 1)[1].strip())["text"] == "hi"

    def test_a_stream_failure_is_reported_in_band(self, chat_client, chat_service):
        """Headers are already sent by the time this can fail, so the error
        has to arrive as an event rather than a status code."""

        async def failing(*args, **kwargs):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover - makes this an async generator

        chat_service.create_chat_stream = failing
        response = chat_client.post("/chat", json={"message": "hi"})
        payload = json.loads(response.text.split("data: ", 1)[1].strip())
        assert payload["type"] == "error" and "model exploded" in payload["message"]

    def test_continuing_a_chat_streams(self, chat_client):
        response = chat_client.post(f"/chat/{CHAT_ID}/continue", json={"message": "m"})
        assert response.status_code == 200

    def test_a_continue_failure_is_reported_in_band(self, chat_client, chat_service):
        async def failing(*args, **kwargs):
            raise RuntimeError("stream died")
            yield  # pragma: no cover

        chat_service.continue_chat_stream = failing
        response = chat_client.post(f"/chat/{CHAT_ID}/continue", json={"message": "m"})
        payload = json.loads(response.text.split("data: ", 1)[1].strip())
        assert payload["type"] == "error"

    def test_an_empty_message_is_rejected(self, chat_client):
        assert chat_client.post("/chat", json={"message": ""}).status_code == 422

    def test_an_over_long_message_is_rejected(self, chat_client):
        assert (
            chat_client.post("/chat", json={"message": "x" * 10001}).status_code == 422
        )


class TestChatReads:
    def test_a_chat_is_returned(self, chat_client):
        assert chat_client.get(f"/chat/{CHAT_ID}").status_code == 200

    def test_the_agent_header_scopes_the_lookup(self, chat_client, chat_service):
        chat_client.get(f"/chat/{CHAT_ID}", headers={"x-agent-id": AGENT_ID})
        assert chat_service.get_chat.call_args.kwargs["agent_id"] == AGENT_ID

    def test_omitting_the_agent_header_searches_the_organization(
        self, chat_client, chat_service
    ):
        chat_client.get(f"/chat/{CHAT_ID}")
        assert chat_service.get_chat.call_args.kwargs["agent_id"] is None

    def test_a_missing_chat_is_a_404(self, chat_client, chat_service):
        chat_service.get_chat.return_value = None
        assert chat_client.get(f"/chat/{CHAT_ID}").status_code == 404

    def test_a_malformed_id_is_a_400(self, chat_client, chat_service):
        chat_service.get_chat.side_effect = ValueError("Invalid chat ID")
        assert chat_client.get("/chat/nope").status_code == 400

    def test_chats_are_listed(self, chat_client):
        response = chat_client.get("/chat")
        assert response.status_code == 200 and len(response.json()) == 1

    def test_the_agent_header_narrows_the_listing(self, chat_client, chat_service):
        chat_client.get("/chat", headers={"x-agent-id": AGENT_ID})
        chat_service.get_all_chats.assert_called_once_with(agent_id=AGENT_ID)


class TestAgentStatus:
    @pytest.fixture
    def status_service(self, monkeypatch) -> MagicMock:
        service = MagicMock(name="agent_status_service")
        service.get_status = AsyncMock(return_value={"status": "running"})
        registry = MagicMock(agent_status_service=service)
        monkeypatch.setattr(
            "app.dependencies.services.get_services", lambda org_id: registry
        )
        return service

    def test_a_status_is_returned(self, chat_client, status_service):
        response = chat_client.get("/chat/status/key-1")
        assert response.status_code == 200 and response.json()["status"] == "running"

    def test_the_status_is_read_for_the_requested_key(
        self, chat_client, status_service
    ):
        chat_client.get("/chat/status/key-1")
        status_service.get_status.assert_awaited_once_with("key-1")

    def test_an_unknown_key_is_a_404(self, chat_client, status_service):
        status_service.get_status = AsyncMock(return_value=None)
        assert chat_client.get("/chat/status/nope").status_code == 404


# --- rag ------------------------------------------------------------------


@pytest.fixture
def retrieval_service() -> MagicMock:
    service = MagicMock(name="retrieval_service")
    service.search = AsyncMock(return_value=[a_rag()])
    return service


@pytest.fixture
def rag_client(make_client, retrieval_service):
    return make_client(
        rag_routes.router, {get_retrieval_service: override(retrieval_service)}
    )


class TestRagSearch:
    def test_search_returns_results(self, rag_client):
        response = rag_client.post("/rag/search", json={"query": "kita"})
        assert response.status_code == 200 and len(response.json()) == 1

    def test_the_query_reaches_the_service(self, rag_client, retrieval_service):
        rag_client.post("/rag/search", json={"query": "kita"})
        assert retrieval_service.search.await_args.kwargs["query"] == "kita"

    def test_the_limit_defaults_to_five(self, rag_client, retrieval_service):
        rag_client.post("/rag/search", json={"query": "kita"})
        assert retrieval_service.search.await_args.kwargs["limit"] == 5

    def test_an_explicit_limit_is_used(self, rag_client, retrieval_service):
        rag_client.post("/rag/search", json={"query": "kita", "limit": 20})
        assert retrieval_service.search.await_args.kwargs["limit"] == 20

    def test_a_null_limit_falls_back_to_the_default(
        self, rag_client, retrieval_service
    ):
        rag_client.post("/rag/search", json={"query": "kita", "limit": None})
        assert retrieval_service.search.await_args.kwargs["limit"] == 5

    def test_a_missing_query_is_rejected(self, rag_client):
        assert rag_client.post("/rag/search", json={}).status_code == 422

    def test_a_search_failure_is_a_500(self, rag_client, retrieval_service):
        retrieval_service.search = AsyncMock(side_effect=RuntimeError("index missing"))
        assert rag_client.post("/rag/search", json={"query": "k"}).status_code == 500

    def test_no_matches_yields_an_empty_list(self, rag_client, retrieval_service):
        retrieval_service.search = AsyncMock(return_value=[])
        assert rag_client.post("/rag/search", json={"query": "k"}).json() == []


# --- events ---------------------------------------------------------------


@pytest.fixture
def event_service() -> MagicMock:
    service = MagicMock(name="event_service")
    service.push = AsyncMock(return_value=None)
    return service


@pytest.fixture
def event_client(make_client, event_service):
    return make_client(
        event_routes.router, {get_event_service: override(event_service)}
    )


class TestPushEvent:
    def valid(self, key: str = "file:completed") -> dict:
        return {
            "event_key": key,
            "payload": {"file_id": "f1", "org_id": ORG_ID},
        }

    @pytest.mark.parametrize("key", ["file:completed", "parse:completed"])
    def test_a_valid_event_is_pushed(self, event_client, key):
        assert (
            event_client.post("/events/push", json=self.valid(key)).status_code == 200
        )

    def test_the_key_and_payload_reach_the_service(self, event_client, event_service):
        event_client.post("/events/push", json=self.valid())
        key, payload = event_service.push.await_args[0]
        assert key == "file:completed"
        assert payload == {"file_id": "f1", "org_id": ORG_ID}

    def test_the_response_echoes_the_event_key(self, event_client):
        body = event_client.post("/events/push", json=self.valid()).json()
        assert body["event"] == "file:completed"

    @pytest.mark.parametrize("key", ["file:completed", "parse:completed"])
    def test_a_payload_missing_required_fields_is_rejected(
        self, event_client, event_service, key
    ):
        """The payload shape is validated against the event key before
        anything reaches Hatchet, so a malformed job never gets queued."""
        response = event_client.post(
            "/events/push", json={"event_key": key, "payload": {"file_id": "f1"}}
        )
        assert response.status_code == 422
        event_service.push.assert_not_awaited()

    def test_an_unknown_event_key_is_rejected(self, event_client, event_service):
        response = event_client.post(
            "/events/push", json={"event_key": "nope:whatever", "payload": {}}
        )
        assert response.status_code == 422
        event_service.push.assert_not_awaited()

    def test_a_missing_payload_is_rejected(self, event_client):
        response = event_client.post(
            "/events/push", json={"event_key": "file:completed"}
        )
        assert response.status_code == 422

    def test_a_broker_failure_is_a_500(self, event_client, event_service):
        event_service.push = AsyncMock(side_effect=RuntimeError("hatchet unreachable"))
        assert event_client.post("/events/push", json=self.valid()).status_code == 500
