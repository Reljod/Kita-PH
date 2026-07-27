"""Tests for app.routes.agent — agent CRUD, chat streaming, memory, tools."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.dependencies import get_agent_rag_service, get_agent_service, get_chat_service
from app.models.agent import AgentResponse
from app.models.chat import ChatResponse
from app.models.rag import RagResponse
from app.routes import agent as agent_routes

from .conftest import override

AGENT_ID = "agent_1"
CHAT_ID = "chat_1"
RAG_ID = "rag_1"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def an_agent(**overrides) -> AgentResponse:
    payload = {
        "id": AGENT_ID,
        "base_id": AGENT_ID,
        "version": 1,
        "name": "Researcher",
        "role": "analyst",
        "goal": "find things",
        "backstory": "trained on the archive",
        "llm_id": "llm_1",
        "tools": [],
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return AgentResponse(**payload)


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


def a_rag(**overrides) -> RagResponse:
    payload = {
        "id": RAG_ID,
        "title": "Note",
        "content": "body",
        "status": "completed",
        "agent_id": AGENT_ID,
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return RagResponse(**payload)


@pytest.fixture
def agent_service() -> MagicMock:
    service = MagicMock(name="agent_service")
    service.create_agent = AsyncMock(return_value=an_agent())
    service.update_agent = AsyncMock(return_value=an_agent())
    service.add_tools = AsyncMock(return_value=True)
    service.remove_tools = AsyncMock(return_value=True)
    service.get_agent.return_value = an_agent()
    service.get_all_agents.return_value = [an_agent()]
    service.delete_agent.return_value = True
    return service


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
def rag_service() -> MagicMock:
    service = MagicMock(name="rag_service")
    service.add_rag = AsyncMock(return_value=a_rag())
    service.edit_rag = AsyncMock(return_value=a_rag())
    service.delete_rag = AsyncMock(return_value=True)
    service.search = AsyncMock(return_value=[a_rag()])
    service.update_embedding = AsyncMock(return_value=None)
    service.get_rag.return_value = a_rag()
    service.get_all_rags.return_value = [a_rag()]
    return service


@pytest.fixture
def client(make_client, agent_service, chat_service, rag_service):
    return make_client(
        agent_routes.router,
        {
            get_agent_service: override(agent_service),
            get_chat_service: override(chat_service),
            get_agent_rag_service: override(rag_service),
        },
    )


def agent_payload(**overrides) -> dict:
    body = {
        "name": "Researcher",
        "role": "analyst",
        "goal": "find things",
        "backstory": "trained on the archive",
        "llm_id": "llm_1",
    }
    body.update(overrides)
    return body


# --- CRUD -----------------------------------------------------------------


class TestCreateAgent:
    def test_an_agent_is_created(self, client):
        assert client.post("/agent/", json=agent_payload()).status_code == 200

    def test_the_request_reaches_the_service(self, client, agent_service):
        client.post("/agent/", json=agent_payload(name="Analyst"))
        assert agent_service.create_agent.await_args[0][0].name == "Analyst"

    def test_a_service_value_error_becomes_a_400(self, client, agent_service):
        agent_service.create_agent = AsyncMock(side_effect=ValueError("bad llm_id"))
        assert client.post("/agent/", json=agent_payload()).status_code == 400

    @pytest.mark.parametrize("field", ["name", "role", "goal", "backstory", "llm_id"])
    def test_a_blank_required_field_is_rejected(self, client, agent_service, field):
        response = client.post("/agent/", json=agent_payload(**{field: ""}))
        assert response.status_code == 422
        agent_service.create_agent.assert_not_awaited()


class TestUpdateAgent:
    def test_an_agent_is_updated(self, client):
        response = client.put(f"/agent/{AGENT_ID}", json={"name": "Renamed"})
        assert response.status_code == 200

    def test_a_new_version_is_cut_by_default(self, client, agent_service):
        client.put(f"/agent/{AGENT_ID}", json={"name": "Renamed"})
        assert agent_service.update_agent.await_args.kwargs["new_version"] is True

    def test_versioning_can_be_turned_off(self, client, agent_service):
        client.put(
            f"/agent/{AGENT_ID}",
            json={"name": "Renamed"},
            params={"new_version": False},
        )
        assert agent_service.update_agent.await_args.kwargs["new_version"] is False

    def test_an_unknown_agent_is_a_404(self, client, agent_service):
        agent_service.update_agent = AsyncMock(return_value=None)
        response = client.put(f"/agent/{AGENT_ID}", json={"name": "Renamed"})
        assert response.status_code == 404

    def test_an_empty_update_is_allowed(self, client):
        assert client.put(f"/agent/{AGENT_ID}", json={}).status_code == 200


class TestGetAgent:
    def test_an_agent_is_returned(self, client):
        assert client.get(f"/agent/{AGENT_ID}").status_code == 200

    def test_an_unknown_agent_is_a_404(self, client, agent_service):
        agent_service.get_agent.return_value = None
        assert client.get(f"/agent/{AGENT_ID}").status_code == 404

    def test_a_versioned_id_is_passed_through_untouched(self, client, agent_service):
        """The service owns the `<base>-v<n>` grammar; the route must not
        pre-parse it."""
        client.get("/agent/agent_1-v3")
        agent_service.get_agent.assert_called_once_with("agent_1-v3")


class TestListAgents:
    def test_agents_are_listed(self, client):
        response = client.get("/agent/")
        assert response.status_code == 200 and len(response.json()) == 1

    def test_last_chat_is_excluded_by_default(self, client, agent_service):
        """Including it costs an extra query per agent, so the list view must
        not opt in silently."""
        client.get("/agent/")
        assert (
            agent_service.get_all_agents.call_args.kwargs["include_last_chat"] is False
        )

    def test_last_chat_can_be_requested(self, client, agent_service):
        client.get("/agent/", params={"last_chat": True})
        assert (
            agent_service.get_all_agents.call_args.kwargs["include_last_chat"] is True
        )

    def test_an_empty_list_is_returned_when_there_are_none(self, client, agent_service):
        agent_service.get_all_agents.return_value = []
        assert client.get("/agent/").json() == []


class TestDeleteAgent:
    def test_an_agent_is_deleted(self, client):
        assert client.delete(f"/agent/{AGENT_ID}").status_code == 200

    def test_an_unknown_agent_is_a_404(self, client, agent_service):
        agent_service.delete_agent.return_value = False
        assert client.delete(f"/agent/{AGENT_ID}").status_code == 404


# --- chat -----------------------------------------------------------------


class TestAgentChatStreaming:
    def test_a_new_chat_streams_events(self, client):
        response = client.post(f"/agent/{AGENT_ID}/chat", json={"message": "hi"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_events_are_server_sent_json(self, client):
        response = client.post(f"/agent/{AGENT_ID}/chat", json={"message": "hi"})
        assert response.text.startswith("data: ")
        assert json.loads(response.text.split("data: ", 1)[1].strip())["text"] == "hi"

    def test_a_stream_failure_is_reported_in_band(self, client, chat_service):
        """The response has already begun, so an error cannot become a status
        code — it has to arrive as an event the client can act on."""

        async def failing(*args, **kwargs):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover - makes this an async generator

        chat_service.create_chat_stream = failing
        response = client.post(f"/agent/{AGENT_ID}/chat", json={"message": "hi"})
        payload = json.loads(response.text.split("data: ", 1)[1].strip())
        assert payload["type"] == "error" and "model exploded" in payload["message"]

    def test_continuing_a_chat_streams_events(self, client):
        response = client.post(
            f"/agent/{AGENT_ID}/chat/{CHAT_ID}/continue", json={"message": "more"}
        )
        assert response.status_code == 200

    def test_an_empty_message_is_rejected(self, client):
        assert (
            client.post(f"/agent/{AGENT_ID}/chat", json={"message": ""}).status_code
            == 422
        )


class TestAgentChatReads:
    def test_a_chat_is_returned(self, client):
        assert client.get(f"/agent/{AGENT_ID}/chat/{CHAT_ID}").status_code == 200

    def test_the_lookup_is_scoped_to_the_agent(self, client, chat_service):
        """Without this, one agent's route could read another agent's chat."""
        client.get(f"/agent/{AGENT_ID}/chat/{CHAT_ID}")
        assert chat_service.get_chat.call_args.kwargs["agent_id"] == AGENT_ID

    def test_a_missing_chat_is_a_404(self, client, chat_service):
        chat_service.get_chat.return_value = None
        assert client.get(f"/agent/{AGENT_ID}/chat/{CHAT_ID}").status_code == 404

    def test_a_malformed_chat_id_is_a_400(self, client, chat_service):
        chat_service.get_chat.side_effect = ValueError("Invalid chat ID")
        assert client.get(f"/agent/{AGENT_ID}/chat/nope").status_code == 400

    def test_chats_are_listed_for_the_agent(self, client, chat_service):
        client.get(f"/agent/{AGENT_ID}/chat")
        assert chat_service.get_all_chats.call_args.kwargs["agent_id"] == AGENT_ID

    def test_preview_mode_is_forwarded(self, client, chat_service):
        client.get(f"/agent/{AGENT_ID}/chat", params={"preview": True})
        assert chat_service.get_all_chats.call_args.kwargs["preview"] is True


# --- agent memory ---------------------------------------------------------


class TestAgentMemory:
    def test_a_memory_is_created(self, client):
        response = client.post(
            f"/agent/{AGENT_ID}/memory", json={"title": "Note", "content": "body"}
        )
        assert response.status_code == 201

    def test_the_memory_is_bound_to_the_agent_in_the_path(self, client, rag_service):
        """A caller must not be able to file a memory against another agent by
        putting a different agent_id in the body."""
        client.post(
            f"/agent/{AGENT_ID}/memory",
            json={"title": "Note", "content": "body", "agent_id": "someone_else"},
        )
        assert rag_service.add_rag.await_args[0][0].agent_id == AGENT_ID

    def test_embedding_is_queued_in_the_background(self, client, rag_service):
        client.post(
            f"/agent/{AGENT_ID}/memory", json={"title": "Note", "content": "body"}
        )
        rag_service.update_embedding.assert_awaited_once_with(RAG_ID)

    def test_a_service_failure_is_a_500(self, client, rag_service):
        rag_service.add_rag = AsyncMock(side_effect=RuntimeError("mongo down"))
        response = client.post(
            f"/agent/{AGENT_ID}/memory", json={"title": "Note", "content": "body"}
        )
        assert response.status_code == 500

    def test_memories_are_listed_for_the_agent(self, client, rag_service):
        client.get(f"/agent/{AGENT_ID}/memory")
        rag_service.get_all_rags.assert_called_once_with(agent_id=AGENT_ID)

    def test_a_memory_is_returned(self, client):
        assert client.get(f"/agent/{AGENT_ID}/memory/{RAG_ID}").status_code == 200

    def test_a_missing_memory_is_a_404(self, client, rag_service):
        rag_service.get_rag.return_value = None
        assert client.get(f"/agent/{AGENT_ID}/memory/{RAG_ID}").status_code == 404

    def test_a_malformed_memory_id_is_a_400(self, client, rag_service):
        rag_service.get_rag.side_effect = ValueError("Invalid id")
        assert client.get(f"/agent/{AGENT_ID}/memory/nope").status_code == 400


class TestAgentMemorySearch:
    def test_search_returns_results(self, client):
        response = client.get(
            f"/agent/{AGENT_ID}/memory/search", params={"query": "kita"}
        )
        assert response.status_code == 200

    def test_the_search_is_scoped_to_the_agent(self, client, rag_service):
        client.get(f"/agent/{AGENT_ID}/memory/search", params={"query": "kita"})
        assert rag_service.search.await_args.kwargs["agent_id"] == AGENT_ID

    def test_the_limit_defaults_to_five(self, client, rag_service):
        client.get(f"/agent/{AGENT_ID}/memory/search", params={"query": "kita"})
        assert rag_service.search.await_args.kwargs["limit"] == 5

    @pytest.mark.parametrize("limit", [0, 51, -1])
    def test_an_out_of_range_limit_is_rejected(self, client, limit):
        response = client.get(
            f"/agent/{AGENT_ID}/memory/search", params={"query": "k", "limit": limit}
        )
        assert response.status_code == 422

    def test_an_empty_query_is_rejected(self, client):
        response = client.get(f"/agent/{AGENT_ID}/memory/search", params={"query": ""})
        assert response.status_code == 422

    def test_a_search_failure_is_a_500(self, client, rag_service):
        rag_service.search = AsyncMock(side_effect=RuntimeError("index missing"))
        response = client.get(
            f"/agent/{AGENT_ID}/memory/search", params={"query": "kita"}
        )
        assert response.status_code == 500


class TestUpdateAgentMemory:
    def test_a_memory_is_updated(self, client):
        response = client.put(
            f"/agent/{AGENT_ID}/memory/{RAG_ID}", json={"title": "Renamed"}
        )
        assert response.status_code == 200

    def test_the_update_is_scoped_to_the_agent(self, client, rag_service):
        client.put(f"/agent/{AGENT_ID}/memory/{RAG_ID}", json={"title": "Renamed"})
        assert rag_service.edit_rag.await_args.kwargs["agent_id"] == AGENT_ID

    def test_re_embedding_is_queued_when_the_content_changed(self, client, rag_service):
        """The service marks the row pending when content changes; that is
        the signal to recompute the vector."""
        rag_service.edit_rag = AsyncMock(return_value=a_rag(status="pending"))
        client.put(f"/agent/{AGENT_ID}/memory/{RAG_ID}", json={"content": "new"})
        rag_service.update_embedding.assert_awaited_once_with(RAG_ID)

    def test_re_embedding_is_skipped_when_nothing_changed(self, client, rag_service):
        rag_service.edit_rag = AsyncMock(return_value=a_rag(status="completed"))
        client.put(f"/agent/{AGENT_ID}/memory/{RAG_ID}", json={"title": "Renamed"})
        rag_service.update_embedding.assert_not_awaited()

    def test_a_missing_memory_is_a_404_not_a_500(self, client, rag_service):
        """The handler raises HTTPException(404) inside a try whose bare
        `except Exception` also catches HTTPException. Re-wrapping it turns
        an ordinary not-found into a server error."""
        rag_service.edit_rag = AsyncMock(return_value=None)
        response = client.put(
            f"/agent/{AGENT_ID}/memory/{RAG_ID}", json={"title": "Renamed"}
        )
        assert response.status_code == 404

    def test_a_malformed_id_is_a_400(self, client, rag_service):
        rag_service.edit_rag = AsyncMock(side_effect=ValueError("Invalid id"))
        response = client.put(
            f"/agent/{AGENT_ID}/memory/nope", json={"title": "Renamed"}
        )
        assert response.status_code == 400

    def test_a_service_failure_is_a_500(self, client, rag_service):
        rag_service.edit_rag = AsyncMock(side_effect=RuntimeError("mongo down"))
        response = client.put(
            f"/agent/{AGENT_ID}/memory/{RAG_ID}", json={"title": "Renamed"}
        )
        assert response.status_code == 500


class TestDeleteAgentMemory:
    def test_a_memory_is_deleted(self, client):
        assert client.delete(f"/agent/{AGENT_ID}/memory/{RAG_ID}").status_code == 200

    def test_the_delete_is_scoped_to_the_agent(self, client, rag_service):
        client.delete(f"/agent/{AGENT_ID}/memory/{RAG_ID}")
        assert rag_service.delete_rag.await_args.kwargs["agent_id"] == AGENT_ID

    def test_a_missing_memory_is_a_404_not_a_500(self, client, rag_service):
        """Same swallowed-HTTPException shape as the update handler."""
        rag_service.delete_rag = AsyncMock(return_value=False)
        assert client.delete(f"/agent/{AGENT_ID}/memory/{RAG_ID}").status_code == 404

    def test_a_malformed_id_is_a_400(self, client, rag_service):
        rag_service.delete_rag = AsyncMock(side_effect=ValueError("Invalid id"))
        assert client.delete(f"/agent/{AGENT_ID}/memory/nope").status_code == 400

    def test_a_service_failure_is_a_500(self, client, rag_service):
        rag_service.delete_rag = AsyncMock(side_effect=RuntimeError("mongo down"))
        assert client.delete(f"/agent/{AGENT_ID}/memory/{RAG_ID}").status_code == 500


# --- tools ----------------------------------------------------------------


class TestAgentTools:
    def test_tools_are_added(self, client):
        response = client.post(
            f"/agent/{AGENT_ID}/tools/add", json={"tool_ids": ["t1"]}
        )
        assert response.status_code == 200

    def test_the_updated_agent_is_returned(self, client, agent_service):
        agent_service.get_agent.return_value = an_agent(tools=["t1"])
        response = client.post(
            f"/agent/{AGENT_ID}/tools/add", json={"tool_ids": ["t1"]}
        )
        assert response.json()["tools"] == ["t1"]

    def test_adding_to_an_unknown_agent_is_a_404(self, client, agent_service):
        agent_service.add_tools = AsyncMock(return_value=False)
        response = client.post(
            f"/agent/{AGENT_ID}/tools/add", json={"tool_ids": ["t1"]}
        )
        assert response.status_code == 404

    def test_an_agent_that_vanishes_mid_request_is_a_404(self, client, agent_service):
        agent_service.get_agent.return_value = None
        response = client.post(
            f"/agent/{AGENT_ID}/tools/add", json={"tool_ids": ["t1"]}
        )
        assert response.status_code == 404

    def test_tools_are_removed(self, client):
        response = client.post(
            f"/agent/{AGENT_ID}/tools/remove", json={"tool_ids": ["t1"]}
        )
        assert response.status_code == 200

    def test_removing_from_an_unknown_agent_is_a_404(self, client, agent_service):
        agent_service.remove_tools = AsyncMock(return_value=False)
        response = client.post(
            f"/agent/{AGENT_ID}/tools/remove", json={"tool_ids": ["t1"]}
        )
        assert response.status_code == 404

    @pytest.mark.parametrize("endpoint", ["add", "remove"])
    def test_an_empty_tool_list_is_rejected(self, client, endpoint):
        response = client.post(
            f"/agent/{AGENT_ID}/tools/{endpoint}", json={"tool_ids": []}
        )
        assert response.status_code == 422
