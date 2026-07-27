"""Tests for app.routes.memory and app.routes.llm.

The memory routes are the org-wide twin of the agent-scoped ones: the agent
filter arrives as an `x-agent-id` header rather than a path segment, and
omitting it means "everything in this organization".
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.dependencies import get_llm_service, get_rag_service
from app.models.llm import LlmResponse
from app.models.rag import RagResponse
from app.routes import llm as llm_routes
from app.routes import memory as memory_routes

from .conftest import override

RAG_ID = "rag_1"
LLM_ID = "64b7f1c2e4b0a1a2b3c4d5e6"
AGENT_ID = "agent_1"
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


def a_rag(**overrides) -> RagResponse:
    payload = {
        "id": RAG_ID,
        "title": "Note",
        "content": "body",
        "status": "completed",
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return RagResponse(**payload)


def an_llm(**overrides) -> LlmResponse:
    payload = {
        "id": LLM_ID,
        "name": "default",
        "model": "x-ai/grok-4.3",
        "provider": "openrouter",
        "created_at": NOW,
        "updated_at": NOW,
    }
    payload.update(overrides)
    return LlmResponse(**payload)


# --- memory ---------------------------------------------------------------


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
def memory_client(make_client, rag_service):
    return make_client(memory_routes.router, {get_rag_service: override(rag_service)})


class TestCreateMemory:
    def test_a_memory_is_created(self, memory_client):
        response = memory_client.post(
            "/memory", json={"title": "Note", "content": "body"}
        )
        assert response.status_code == 201

    def test_embedding_is_queued_in_the_background(self, memory_client, rag_service):
        memory_client.post("/memory", json={"title": "Note", "content": "body"})
        rag_service.update_embedding.assert_awaited_once_with(RAG_ID)

    def test_a_service_failure_is_a_500(self, memory_client, rag_service):
        rag_service.add_rag = AsyncMock(side_effect=RuntimeError("mongo down"))
        response = memory_client.post(
            "/memory", json={"title": "Note", "content": "body"}
        )
        assert response.status_code == 500

    @pytest.mark.parametrize(
        "body",
        [
            {"title": "", "content": "body"},
            {"title": "Note", "content": ""},
            {"title": "Note"},
            {"content": "body"},
            {"title": "x" * 201, "content": "body"},
        ],
    )
    def test_invalid_input_is_rejected(self, memory_client, rag_service, body):
        assert memory_client.post("/memory", json=body).status_code == 422
        rag_service.add_rag.assert_not_awaited()


class TestListMemories:
    def test_memories_are_listed(self, memory_client):
        response = memory_client.get("/memory")
        assert response.status_code == 200 and len(response.json()) == 1

    def test_omitting_the_agent_header_lists_the_whole_organization(
        self, memory_client, rag_service
    ):
        memory_client.get("/memory")
        rag_service.get_all_rags.assert_called_once_with(agent_id=None)

    def test_the_agent_header_narrows_the_listing(self, memory_client, rag_service):
        memory_client.get("/memory", headers={"x-agent-id": AGENT_ID})
        rag_service.get_all_rags.assert_called_once_with(agent_id=AGENT_ID)


class TestSearchMemory:
    def test_search_returns_results(self, memory_client):
        response = memory_client.get("/memory/search", params={"query": "kita"})
        assert response.status_code == 200

    def test_the_agent_header_scopes_the_search(self, memory_client, rag_service):
        memory_client.get(
            "/memory/search", params={"query": "kita"}, headers={"x-agent-id": AGENT_ID}
        )
        assert rag_service.search.await_args.kwargs["agent_id"] == AGENT_ID

    def test_the_limit_defaults_to_five(self, memory_client, rag_service):
        memory_client.get("/memory/search", params={"query": "kita"})
        assert rag_service.search.await_args.kwargs["limit"] == 5

    @pytest.mark.parametrize("limit", [0, 51])
    def test_an_out_of_range_limit_is_rejected(self, memory_client, limit):
        response = memory_client.get(
            "/memory/search", params={"query": "k", "limit": limit}
        )
        assert response.status_code == 422

    def test_an_empty_query_is_rejected(self, memory_client):
        assert (
            memory_client.get("/memory/search", params={"query": ""}).status_code == 422
        )

    def test_a_search_failure_is_a_500(self, memory_client, rag_service):
        rag_service.search = AsyncMock(side_effect=RuntimeError("index missing"))
        response = memory_client.get("/memory/search", params={"query": "kita"})
        assert response.status_code == 500


class TestGetMemory:
    def test_a_memory_is_returned(self, memory_client):
        assert memory_client.get(f"/memory/{RAG_ID}").status_code == 200

    def test_a_missing_memory_is_a_404(self, memory_client, rag_service):
        rag_service.get_rag.return_value = None
        assert memory_client.get(f"/memory/{RAG_ID}").status_code == 404

    def test_a_malformed_id_is_a_400(self, memory_client, rag_service):
        rag_service.get_rag.side_effect = ValueError("Invalid id")
        assert memory_client.get("/memory/nope").status_code == 400


class TestUpdateMemory:
    def test_a_memory_is_updated(self, memory_client):
        response = memory_client.put(f"/memory/{RAG_ID}", json={"title": "Renamed"})
        assert response.status_code == 200

    def test_re_embedding_is_queued_when_the_content_changed(
        self, memory_client, rag_service
    ):
        rag_service.edit_rag = AsyncMock(return_value=a_rag(status="pending"))
        memory_client.put(f"/memory/{RAG_ID}", json={"content": "new"})
        rag_service.update_embedding.assert_awaited_once_with(RAG_ID)

    def test_re_embedding_is_skipped_when_nothing_changed(
        self, memory_client, rag_service
    ):
        rag_service.edit_rag = AsyncMock(return_value=a_rag(status="completed"))
        memory_client.put(f"/memory/{RAG_ID}", json={"title": "Renamed"})
        rag_service.update_embedding.assert_not_awaited()

    def test_a_missing_memory_is_a_404_not_a_500(self, memory_client, rag_service):
        """Same swallowed-HTTPException shape as the agent memory routes."""
        rag_service.edit_rag = AsyncMock(return_value=None)
        response = memory_client.put(f"/memory/{RAG_ID}", json={"title": "Renamed"})
        assert response.status_code == 404

    def test_a_malformed_id_is_a_400(self, memory_client, rag_service):
        rag_service.edit_rag = AsyncMock(side_effect=ValueError("Invalid id"))
        assert memory_client.put("/memory/nope", json={"title": "R"}).status_code == 400

    def test_a_service_failure_is_a_500(self, memory_client, rag_service):
        rag_service.edit_rag = AsyncMock(side_effect=RuntimeError("mongo down"))
        response = memory_client.put(f"/memory/{RAG_ID}", json={"title": "Renamed"})
        assert response.status_code == 500


class TestDeleteMemory:
    def test_a_memory_is_deleted(self, memory_client):
        assert memory_client.delete(f"/memory/{RAG_ID}").status_code == 200

    def test_the_agent_header_scopes_the_delete(self, memory_client, rag_service):
        memory_client.delete(f"/memory/{RAG_ID}", headers={"x-agent-id": AGENT_ID})
        assert rag_service.delete_rag.await_args.kwargs["agent_id"] == AGENT_ID

    def test_a_missing_memory_is_a_404_not_a_500(self, memory_client, rag_service):
        rag_service.delete_rag = AsyncMock(return_value=False)
        assert memory_client.delete(f"/memory/{RAG_ID}").status_code == 404

    def test_a_malformed_id_is_a_400(self, memory_client, rag_service):
        rag_service.delete_rag = AsyncMock(side_effect=ValueError("Invalid id"))
        assert memory_client.delete("/memory/nope").status_code == 400

    def test_a_service_failure_is_a_500(self, memory_client, rag_service):
        rag_service.delete_rag = AsyncMock(side_effect=RuntimeError("mongo down"))
        assert memory_client.delete(f"/memory/{RAG_ID}").status_code == 500


# --- llm ------------------------------------------------------------------


@pytest.fixture
def llm_service() -> MagicMock:
    service = MagicMock(name="llm_service")
    service.add_llm.return_value = an_llm()
    service.list_llms.return_value = [an_llm()]
    service.delete_llm.return_value = True
    return service


@pytest.fixture
def llm_client(make_client, llm_service):
    return make_client(llm_routes.router, {get_llm_service: override(llm_service)})


class TestLlmRoutes:
    def test_an_llm_is_registered(self, llm_client):
        response = llm_client.post(
            "/llm/", json={"name": "fast", "model": "x-ai/grok-4.3"}
        )
        assert response.status_code == 200

    def test_the_request_reaches_the_service(self, llm_client, llm_service):
        llm_client.post("/llm/", json={"name": "fast", "model": "m"})
        assert llm_service.add_llm.call_args[0][0].name == "fast"

    def test_the_provider_defaults_to_openrouter(self, llm_client, llm_service):
        llm_client.post("/llm/", json={"name": "fast", "model": "m"})
        assert llm_service.add_llm.call_args[0][0].provider == "openrouter"

    @pytest.mark.parametrize(
        "body", [{"name": "", "model": "m"}, {"name": "f", "model": ""}, {"name": "f"}]
    )
    def test_invalid_input_is_rejected(self, llm_client, llm_service, body):
        assert llm_client.post("/llm/", json=body).status_code == 422
        llm_service.add_llm.assert_not_called()

    def test_llms_are_listed(self, llm_client):
        response = llm_client.get("/llm/")
        assert response.status_code == 200 and len(response.json()) == 1

    def test_an_empty_registry_lists_nothing(self, llm_client, llm_service):
        llm_service.list_llms.return_value = []
        assert llm_client.get("/llm/").json() == []

    def test_an_llm_is_deleted(self, llm_client):
        assert llm_client.delete(f"/llm/{LLM_ID}").status_code == 200

    def test_a_missing_llm_is_a_404(self, llm_client, llm_service):
        """delete_llm only catches ValueError, so unlike the memory and tool
        routes its 404 was never at risk of being re-wrapped as a 500."""
        llm_service.delete_llm.return_value = False
        assert llm_client.delete(f"/llm/{LLM_ID}").status_code == 404

    def test_a_malformed_id_is_a_400(self, llm_client, llm_service):
        llm_service.delete_llm.side_effect = ValueError("Invalid LLM ID")
        assert llm_client.delete("/llm/nope").status_code == 400
