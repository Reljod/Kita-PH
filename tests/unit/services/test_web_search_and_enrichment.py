"""Tests for SerperSearchService and RagEnrichmentService.

Both wrap a paid third-party call. The web search deliberately re-raises so
its caller can decide what to tell the user; enrichment deliberately does not,
because a memory that fails to enrich should still get an embedding and stay
searchable rather than being lost.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db import TenantCollection
from app.services import rag_enrichment_service as enrichment_module
from app.services.rag_enrichment_service import RagEnrichmentService
from app.services.web_search_service import SerperSearchService

ORG_ID = "org_test_0001"


def a_response(payload: dict, status_error: Exception | None = None):
    response = MagicMock()
    response.json.return_value = payload
    if status_error:
        response.raise_for_status.side_effect = status_error
    return response


@pytest.fixture
def serper_post(monkeypatch):
    """Capture the outbound Serper request without making one."""
    captured = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return captured.get("response") or a_response({"organic": []})

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: Client())
    return captured


# --- web search -----------------------------------------------------------


class TestSerperSearch:
    async def test_results_are_returned(self, serper_post):
        serper_post["response"] = a_response({"organic": [{"title": "T"}]})
        result = await SerperSearchService().search("kita")
        assert result["organic"][0]["title"] == "T"

    async def test_a_missing_key_is_rejected_before_the_call(
        self, monkeypatch, serper_post
    ):
        """Calling without one just returns a 403 that costs a round trip and
        reads as a search failure rather than a configuration one."""
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SERPER_API_KEY"):
            await SerperSearchService().search("kita")
        assert "payload" not in serper_post

    async def test_the_key_is_sent_as_a_header(self, serper_post):
        await SerperSearchService().search("kita")
        assert serper_post["headers"]["X-API-KEY"]

    async def test_the_query_and_options_are_sent(self, serper_post):
        await SerperSearchService().search(
            "kita", country="ph", language="tl", page=2, search_type="news"
        )
        payload = serper_post["payload"]
        assert payload["q"] == "kita"
        assert payload["gl"] == "ph"
        assert payload["hl"] == "tl"
        assert payload["page"] == 2
        assert payload["type"] == "news"

    @pytest.mark.parametrize(
        "friendly,expected",
        [
            ("past_hour", "qdr:h"),
            ("past_24_hours", "qdr:d"),
            ("past_week", "qdr:w"),
            ("past_month", "qdr:m"),
            ("past_year", "qdr:y"),
        ],
    )
    async def test_friendly_date_ranges_are_translated(
        self, serper_post, friendly, expected
    ):
        """The agent picks these by name; Serper only understands its own
        tbs codes."""
        await SerperSearchService().search("kita", date_range=friendly)
        assert serper_post["payload"]["tbs"] == expected

    async def test_a_raw_tbs_code_passes_through(self, serper_post):
        await SerperSearchService().search("kita", date_range="qdr:h")
        assert serper_post["payload"]["tbs"] == "qdr:h"

    async def test_no_date_range_sends_no_filter(self, serper_post):
        await SerperSearchService().search("kita")
        assert "tbs" not in serper_post["payload"]

    async def test_an_http_error_is_re_raised(self, serper_post):
        """The caller decides what to tell the user; swallowing it here would
        make an outage look like a query with no results."""
        serper_post["response"] = a_response({}, status_error=RuntimeError("403"))
        with pytest.raises(RuntimeError):
            await SerperSearchService().search("kita")

    async def test_a_non_dict_response_is_still_returned(self, serper_post):
        serper_post["response"] = a_response([])
        assert await SerperSearchService().search("kita") == []


# --- memory enrichment ----------------------------------------------------


@pytest.fixture
def rag_collection(mongo_db):
    return mongo_db["rag"]


@pytest.fixture
def service(rag_collection, monkeypatch) -> RagEnrichmentService:
    monkeypatch.setattr(
        enrichment_module, "_create_embedding", AsyncMock(return_value=[0.1] * 8)
    )
    # The model is only constructed lazily; stub it so no provider is built.
    service = RagEnrichmentService(TenantCollection(rag_collection, ORG_ID))
    monkeypatch.setattr(service, "_get_llm_model", lambda: MagicMock())
    return service


@pytest.fixture
def enriching_agent(monkeypatch):
    """Replace the pydantic-ai Agent with one returning a fixed enrichment."""

    def make(question="What is X?", answer="It is Y.", fail=False):
        agent = MagicMock()
        if fail:
            agent.run = AsyncMock(side_effect=RuntimeError("model refused"))
        else:
            agent.run = AsyncMock(
                return_value=SimpleNamespace(
                    output=SimpleNamespace(question=question, answer=answer)
                )
            )
        monkeypatch.setattr(enrichment_module, "Agent", lambda **kw: agent)
        return agent

    return make


def seed(collection, **overrides):
    now = datetime.now(timezone.utc)
    doc = {
        "org_id": ORG_ID,
        "title": "Deploy runbook",
        "content": "Run make deploy",
        "original_content": "Run make deploy",
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    doc.update(overrides)
    return collection.insert_one(doc).inserted_id


class TestEnrichAndEmbed:
    async def test_a_missing_document_yields_nothing(self, service):
        from bson import ObjectId

        assert await service.enrich_and_embed(str(ObjectId())) is None

    async def test_a_malformed_id_is_rejected(self, service):
        with pytest.raises(ValueError):
            await service.enrich_and_embed("nope")

    async def test_the_generated_question_is_stored(
        self, service, rag_collection, enriching_agent
    ):
        enriching_agent(question="How do I deploy?")
        rag_id = seed(rag_collection)
        await service.enrich_and_embed(str(rag_id))
        assert rag_collection.find_one({})["question"] == "How do I deploy?"

    async def test_the_answer_replaces_the_content(
        self, service, rag_collection, enriching_agent
    ):
        """Retrieval returns `content` to the model, so the distilled answer
        is what should reach it -- the raw note stays in original_content."""
        enriching_agent(answer="Run make deploy.")
        rag_id = seed(rag_collection)
        await service.enrich_and_embed(str(rag_id))
        stored = rag_collection.find_one({})
        assert stored["content"] == "Run make deploy."
        assert stored["original_content"] == "Run make deploy"

    async def test_the_entry_becomes_searchable(
        self, service, rag_collection, enriching_agent
    ):
        enriching_agent()
        rag_id = seed(rag_collection)
        await service.enrich_and_embed(str(rag_id))
        stored = rag_collection.find_one({})
        assert stored["status"] == "completed" and stored["embedding"]

    async def test_the_question_is_what_gets_embedded(
        self, service, rag_collection, enriching_agent, monkeypatch
    ):
        """Embedding the question rather than the note is the whole point:
        an incoming query is a question, so the vectors match shape."""
        embed = AsyncMock(return_value=[0.1] * 8)
        monkeypatch.setattr(enrichment_module, "_create_embedding", embed)
        enriching_agent(question="How do I deploy?")
        rag_id = seed(rag_collection)
        await service.enrich_and_embed(str(rag_id))
        assert "How do I deploy?" in embed.await_args[0][0]

    async def test_the_date_is_embedded_alongside_the_question(
        self, service, rag_collection, enriching_agent, monkeypatch
    ):
        """Two notes on the same topic a year apart are otherwise
        indistinguishable to the retriever."""
        embed = AsyncMock(return_value=[0.1] * 8)
        monkeypatch.setattr(enrichment_module, "_create_embedding", embed)
        enriching_agent()
        created = datetime(2026, 3, 14, tzinfo=timezone.utc)
        rag_id = seed(rag_collection, created_at=created)
        await service.enrich_and_embed(str(rag_id))
        assert "2026-03-14" in embed.await_args[0][0]

    async def test_the_original_note_is_what_gets_enriched(
        self, service, rag_collection, enriching_agent
    ):
        """Re-enriching has to start from the user's words, not from the
        previous run's distilled answer, or it drifts further each time."""
        agent = enriching_agent()
        rag_id = seed(
            rag_collection, content="already distilled", original_content="the source"
        )
        await service.enrich_and_embed(str(rag_id))
        assert "the source" in agent.run.await_args[0][0]

    async def test_a_note_without_an_original_falls_back_to_its_content(
        self, service, rag_collection, enriching_agent
    ):
        agent = enriching_agent()
        rag_id = seed(rag_collection, original_content=None, content="just content")
        await service.enrich_and_embed(str(rag_id))
        assert "just content" in agent.run.await_args[0][0]

    async def test_a_model_failure_still_produces_a_searchable_entry(
        self, service, rag_collection, enriching_agent
    ):
        """Losing enrichment is acceptable; losing the memory is not."""
        enriching_agent(fail=True)
        rag_id = seed(rag_collection)
        await service.enrich_and_embed(str(rag_id))
        stored = rag_collection.find_one({})
        assert stored["status"] == "completed" and stored["embedding"]

    async def test_the_fallback_question_names_the_entry(
        self, service, rag_collection, enriching_agent
    ):
        enriching_agent(fail=True)
        rag_id = seed(rag_collection, title="Deploy runbook")
        await service.enrich_and_embed(str(rag_id))
        assert "Deploy runbook" in rag_collection.find_one({})["question"]

    async def test_the_fallback_answer_is_the_note_itself(
        self, service, rag_collection, enriching_agent
    ):
        enriching_agent(fail=True)
        rag_id = seed(rag_collection, original_content="Run make deploy")
        await service.enrich_and_embed(str(rag_id))
        assert rag_collection.find_one({})["answer"] == "Run make deploy"

    async def test_another_organizations_entry_is_not_enriched(
        self, rag_collection, monkeypatch, enriching_agent
    ):
        monkeypatch.setattr(
            enrichment_module, "_create_embedding", AsyncMock(return_value=[0.1] * 8)
        )
        enriching_agent()
        rag_id = seed(rag_collection, org_id="org_other")
        service = RagEnrichmentService(TenantCollection(rag_collection, ORG_ID))
        assert await service.enrich_and_embed(str(rag_id)) is None
