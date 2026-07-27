"""Tests for the agent-status WebSocket in app.routes.chat.

A WebSocket handshake bypasses the HTTP middleware stack, so this endpoint
authenticates itself from a query parameter. That makes it the one place
where an auth mistake would not be caught by the usual dependency chain —
hence the emphasis on the rejection paths.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.auth import TokenData
from app.routes import chat as chat_routes

ORG_ID = "org_abc"
USER_ID = "user_1"
STATUS_KEY = "key-1"


def pubsub_yielding(*messages) -> MagicMock:
    """A stub Redis pubsub whose listen() replays the given messages."""
    pubsub = MagicMock(name="pubsub")
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.close = AsyncMock()

    async def listen():
        for message in messages:
            yield message

    pubsub.listen = listen
    return pubsub


@pytest.fixture
def status_service() -> MagicMock:
    service = MagicMock(name="agent_status_service")
    service.get_status = AsyncMock(return_value={"status": "running", "step": "draft"})
    service._get_channel_name = MagicMock(return_value="chan:key-1")
    service.redis = MagicMock()
    service.redis.pubsub = MagicMock(return_value=pubsub_yielding())
    return service


@pytest.fixture
def wired(monkeypatch, status_service):
    """Authenticate successfully and hand back the stubbed status service."""
    auth = MagicMock()
    auth.verify_token.return_value = TokenData(user_id=USER_ID, org_id=ORG_ID)
    monkeypatch.setattr(chat_routes, "AuthService", lambda: auth)
    registry = MagicMock(agent_status_service=status_service)
    monkeypatch.setattr(
        "app.dependencies.services.get_services", lambda org_id: registry
    )
    return status_service


@pytest.fixture
def client(make_client):
    return make_client(chat_routes.router)


def connect(client, key: str = STATUS_KEY, token: str = "good-token"):
    return client.websocket_connect(f"/chat/status/ws/{key}?token={token}")


# --- authentication -------------------------------------------------------


class TestWebSocketAuth:
    def test_an_invalid_token_is_refused(self, client, monkeypatch):
        auth = MagicMock()
        auth.verify_token.return_value = None
        monkeypatch.setattr(chat_routes, "AuthService", lambda: auth)

        # The endpoint closes without accepting, which surfaces to the client
        # as a failed handshake rather than an open socket.
        with pytest.raises(Exception):
            with connect(client, token="bad-token"):
                pass

    def test_a_token_without_an_organization_is_refused(self, client, monkeypatch):
        """An org-less token authenticates a user but grants no tenant, so it
        must not open a stream scoped to one."""
        auth = MagicMock()
        auth.verify_token.return_value = TokenData(user_id=USER_ID, org_id=None)
        monkeypatch.setattr(chat_routes, "AuthService", lambda: auth)

        with pytest.raises(Exception):
            with connect(client):
                pass

    def test_the_query_token_is_what_gets_verified(self, client, monkeypatch):
        auth = MagicMock()
        auth.verify_token.return_value = None
        monkeypatch.setattr(chat_routes, "AuthService", lambda: auth)

        with pytest.raises(Exception):
            with connect(client, token="the-token"):
                pass
        auth.verify_token.assert_called_once_with("the-token")

    def test_a_missing_token_is_a_validation_failure(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect(f"/chat/status/ws/{STATUS_KEY}"):
                pass


# --- streaming ------------------------------------------------------------


class TestWebSocketStreaming:
    def test_the_current_status_is_sent_on_connect(self, client, wired):
        """A client joining mid-run must not have to wait for the next
        publish to learn where things stand."""
        with connect(client) as ws:
            assert ws.receive_json() == {"status": "running", "step": "draft"}

    def test_no_initial_frame_when_there_is_no_status_yet(self, client, wired):
        wired.get_status = AsyncMock(return_value=None)
        wired.redis.pubsub = MagicMock(
            return_value=pubsub_yielding(
                {"type": "message", "data": json.dumps({"status": "completed"})}
            )
        )
        with connect(client) as ws:
            assert ws.receive_json() == {"status": "completed"}

    def test_published_updates_are_forwarded(self, client, wired):
        wired.get_status = AsyncMock(return_value=None)
        wired.redis.pubsub = MagicMock(
            return_value=pubsub_yielding(
                {"type": "message", "data": json.dumps({"status": "running"})},
                {"type": "message", "data": json.dumps({"status": "completed"})},
            )
        )
        with connect(client) as ws:
            assert ws.receive_json()["status"] == "running"
            assert ws.receive_json()["status"] == "completed"

    def test_non_message_frames_are_ignored(self, client, wired):
        """Redis emits a 'subscribe' confirmation frame that is not payload."""
        wired.get_status = AsyncMock(return_value=None)
        wired.redis.pubsub = MagicMock(
            return_value=pubsub_yielding(
                {"type": "subscribe", "data": 1},
                {"type": "message", "data": json.dumps({"status": "completed"})},
            )
        )
        with connect(client) as ws:
            assert ws.receive_json() == {"status": "completed"}

    @pytest.mark.parametrize("terminal", ["completed", "failed"])
    def test_a_terminal_status_ends_the_stream(self, client, wired, terminal):
        """Otherwise the socket would linger after the run is over."""
        wired.get_status = AsyncMock(return_value=None)
        wired.redis.pubsub = MagicMock(
            return_value=pubsub_yielding(
                {"type": "message", "data": json.dumps({"status": terminal})},
                {
                    "type": "message",
                    "data": json.dumps({"status": "should-not-arrive"}),
                },
            )
        )
        with connect(client) as ws:
            assert ws.receive_json()["status"] == terminal

    def test_the_channel_subscribed_to_is_derived_from_the_status_key(
        self, client, wired
    ):
        pubsub = pubsub_yielding()
        wired.redis.pubsub = MagicMock(return_value=pubsub)
        with connect(client) as ws:
            ws.receive_json()
        wired._get_channel_name.assert_called_once_with(STATUS_KEY)
        pubsub.subscribe.assert_awaited_once_with("chan:key-1")

    def test_the_subscription_is_released_when_the_stream_ends(self, client, wired):
        """A leaked subscription keeps a Redis connection busy for every run
        that ever streamed."""
        pubsub = pubsub_yielding(
            {"type": "message", "data": json.dumps({"status": "completed"})}
        )
        wired.get_status = AsyncMock(return_value=None)
        wired.redis.pubsub = MagicMock(return_value=pubsub)
        with connect(client) as ws:
            ws.receive_json()
        pubsub.unsubscribe.assert_awaited_once_with("chan:key-1")
        pubsub.close.assert_awaited_once()

    def test_a_backend_failure_closes_cleanly_rather_than_hanging(self, client, wired):
        wired.redis.pubsub = MagicMock(side_effect=RuntimeError("redis gone"))
        with connect(client) as ws:
            ws.receive_json()  # the initial status still lands
        # Reaching here without an exception means the handler closed the
        # socket instead of leaving the client waiting.
