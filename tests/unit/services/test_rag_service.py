"""Tests for app.services.rag_service.

Two things here are load-bearing. The agent filter decides which memories a
caller can see, and its `agent_id` comes straight from a request header --
so escaping matters as much as the matching does. And `search` has to keep
working when the vector index is empty or the embedding call fails, because
memory that silently returns nothing looks identical to memory that is empty.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db import TenantCollection
from app.exceptions import MemoryNotFoundError
from app.models.rag import RagCreateRequest, RagUpdateRequest
from app.services import rag_service as rag_module
from app.services.rag_service import MongoVectorDbRagService, format_rag_response

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"
AGENT_A = "6a6700000000000000000001"
AGENT_B = "6a6700000000000000000002"


@pytest.fixture
def rag_collection(mongo_db):
    return mongo_db["rag"]


@pytest.fixture
def service(rag_collection) -> MongoVectorDbRagService:
    return MongoVectorDbRagService(TenantCollection(rag_collection, ORG_ID))


@pytest.fixture
def no_embeddings(monkeypatch):
    """Embedding and rerank both go out over the network; stub them off."""
    monkeypatch.setattr(
        rag_module, "_create_embedding", AsyncMock(return_value=[0.1] * 8)
    )
    monkeypatch.setattr(rag_module, "_rerank", AsyncMock(return_value=[]))


def seed(collection, *, agent_id=None, title="Note", content="body", **extra):
    now = datetime.now(timezone.utc)
    doc = {
        "org_id": ORG_ID,
        "agent_id": agent_id,
        "title": title,
        "content": content,
        "original_content": content,
        "status": "completed",
        "created_at": now,
        "updated_at": now,
    }
    doc.update(extra)
    return collection.insert_one(doc).inserted_id


# --- the agent filter -----------------------------------------------------


class TestAgentFilter:
    def test_no_agent_means_no_filter(self, service):
        assert service._get_agent_filter(None) == {}

    def test_an_empty_agent_means_no_filter(self, service):
        assert service._get_agent_filter("") == {}

    def test_a_versioned_id_matches_its_base(self, service, rag_collection):
        """Memories are attached to the base agent, so a caller arriving with
        a pinned id must still see them."""
        seed(rag_collection, agent_id=AGENT_A, title="mine")
        found = service.get_all_rags(agent_id=f"{AGENT_A}-v3")
        assert [r.title for r in found] == ["mine"]

    def test_a_wildcard_agent_id_does_not_match_every_agent(
        self, service, rag_collection
    ):
        """The id is interpolated into a $regex and arrives from the client's
        x-agent-id header. Unescaped, ".*" became "^.*(-v\\d+)?$" -- a filter
        matching every agent, leaking every agent's memories in the org."""
        seed(rag_collection, agent_id=AGENT_A, title="A private")
        seed(rag_collection, agent_id=AGENT_B, title="B private")
        assert [r.title for r in service.get_all_rags(agent_id=".*")] == []

    @pytest.mark.parametrize("hostile", [".*", "^.*$", "a|b", "[a-z]+", "(a+)+"])
    def test_regex_metacharacters_are_treated_as_literals(
        self, service, rag_collection, hostile
    ):
        seed(rag_collection, agent_id=AGENT_A, title="A private")
        assert service.get_all_rags(agent_id=hostile) == []

    def test_an_agent_sees_its_own_memories(self, service, rag_collection):
        seed(rag_collection, agent_id=AGENT_A, title="mine")
        seed(rag_collection, agent_id=AGENT_B, title="theirs")
        assert [r.title for r in service.get_all_rags(agent_id=AGENT_A)] == ["mine"]

    def test_an_agent_also_sees_organization_wide_memories(
        self, service, rag_collection
    ):
        seed(rag_collection, agent_id=None, title="shared")
        assert [r.title for r in service.get_all_rags(agent_id=AGENT_A)] == ["shared"]


# --- create ---------------------------------------------------------------


class TestAddRag:
    async def test_a_memory_is_created(self, service):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert created.title == "T" and created.content == "C"

    async def test_a_new_memory_is_pending(self, service):
        """Embedding happens afterwards; "completed" before that would make
        it searchable with no vector."""
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert created.status == "pending"

    async def test_the_original_content_is_kept(self, service, rag_collection):
        """Enrichment rewrites `content`, so the user's own words have to be
        preserved to re-enrich from later."""
        await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert rag_collection.find_one({})["original_content"] == "C"

    async def test_it_is_scoped_to_the_organization(self, service, rag_collection):
        await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert rag_collection.find_one({})["org_id"] == ORG_ID

    async def test_a_versioned_agent_id_is_stored_as_its_base(
        self, service, rag_collection
    ):
        """Otherwise a memory added under -v2 would vanish the moment the
        agent moved to -v3."""
        await service.add_rag(
            RagCreateRequest(title="T", content="C", agent_id=f"{AGENT_A}-v2")
        )
        assert rag_collection.find_one({})["agent_id"] == AGENT_A

    async def test_a_memory_can_be_organization_wide(self, service, rag_collection):
        await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert rag_collection.find_one({})["agent_id"] is None


# --- read -----------------------------------------------------------------


class TestGetRag:
    def test_a_memory_is_returned(self, service, rag_collection):
        rag_id = seed(rag_collection)
        assert service.get_rag(str(rag_id)).id == str(rag_id)

    def test_a_missing_memory_raises(self, service):
        from bson import ObjectId

        with pytest.raises(MemoryNotFoundError):
            service.get_rag(str(ObjectId()))

    def test_a_malformed_id_raises_rather_than_leaking_a_bson_error(self, service):
        """The id comes from the URL, so a bad one is a 404, not a 500."""
        with pytest.raises(MemoryNotFoundError):
            service.get_rag("not-an-object-id")

    def test_another_agents_memory_is_not_readable(self, service, rag_collection):
        rag_id = seed(rag_collection, agent_id=AGENT_B)
        with pytest.raises(MemoryNotFoundError):
            service.get_rag(str(rag_id), agent_id=AGENT_A)

    def test_an_organization_wide_memory_is_readable_by_any_agent(
        self, service, rag_collection
    ):
        rag_id = seed(rag_collection, agent_id=None)
        assert service.get_rag(str(rag_id), agent_id=AGENT_A)

    def test_another_organization_cannot_read_it(self, rag_collection):
        rag_id = seed(rag_collection)
        intruder = MongoVectorDbRagService(TenantCollection(rag_collection, OTHER_ORG))
        with pytest.raises(MemoryNotFoundError):
            intruder.get_rag(str(rag_id))


class TestGetAllRags:
    def test_no_memories_yields_an_empty_list(self, service):
        assert service.get_all_rags() == []

    def test_every_memory_is_listed_without_a_filter(self, service, rag_collection):
        seed(rag_collection, agent_id=AGENT_A)
        seed(rag_collection, agent_id=None)
        assert len(service.get_all_rags()) == 2

    def test_the_newest_comes_first(self, service, rag_collection):
        old = datetime.now(timezone.utc) - timedelta(days=5)
        seed(rag_collection, title="old", updated_at=old)
        seed(rag_collection, title="new")
        assert [r.title for r in service.get_all_rags()] == ["new", "old"]

    def test_another_organizations_memories_are_not_listed(self, rag_collection):
        seed(rag_collection)
        intruder = MongoVectorDbRagService(TenantCollection(rag_collection, OTHER_ORG))
        assert intruder.get_all_rags() == []


# --- update / delete ------------------------------------------------------


class TestEditRag:
    async def test_the_title_is_updated(self, service, rag_collection):
        rag_id = seed(rag_collection)
        updated = await service.edit_rag(str(rag_id), RagUpdateRequest(title="New"))
        assert updated.title == "New"

    async def test_editing_the_content_requeues_the_embedding(
        self, service, rag_collection
    ):
        """A stale vector would keep matching the old text."""
        rag_id = seed(rag_collection)
        updated = await service.edit_rag(str(rag_id), RagUpdateRequest(content="new"))
        assert updated.status == "pending"

    async def test_editing_the_content_resets_the_original(
        self, service, rag_collection
    ):
        rag_id = seed(rag_collection)
        await service.edit_rag(str(rag_id), RagUpdateRequest(content="new"))
        assert rag_collection.find_one({"_id": rag_id})["original_content"] == "new"

    async def test_editing_only_the_title_leaves_the_embedding_alone(
        self, service, rag_collection
    ):
        rag_id = seed(rag_collection)
        updated = await service.edit_rag(str(rag_id), RagUpdateRequest(title="New"))
        assert updated.status == "completed"

    async def test_an_empty_update_is_a_no_op(self, service, rag_collection):
        rag_id = seed(rag_collection, title="Original")
        updated = await service.edit_rag(str(rag_id), RagUpdateRequest())
        assert updated.title == "Original"

    async def test_a_missing_memory_raises(self, service):
        from bson import ObjectId

        with pytest.raises(MemoryNotFoundError):
            await service.edit_rag(str(ObjectId()), RagUpdateRequest(title="x"))

    async def test_a_malformed_id_raises(self, service):
        with pytest.raises(MemoryNotFoundError):
            await service.edit_rag("nope", RagUpdateRequest(title="x"))

    async def test_another_agent_cannot_edit_it(self, service, rag_collection):
        rag_id = seed(rag_collection, agent_id=AGENT_B)
        with pytest.raises(MemoryNotFoundError):
            await service.edit_rag(
                str(rag_id), RagUpdateRequest(title="x"), agent_id=AGENT_A
            )

    async def test_a_wildcard_agent_cannot_edit_another_agents_memory(
        self, service, rag_collection
    ):
        rag_id = seed(rag_collection, agent_id=AGENT_B)
        with pytest.raises(MemoryNotFoundError):
            await service.edit_rag(
                str(rag_id), RagUpdateRequest(title="x"), agent_id=".*"
            )


class TestDeleteRag:
    async def test_a_memory_is_deleted(self, service, rag_collection):
        rag_id = seed(rag_collection)
        assert await service.delete_rag(str(rag_id)) is True
        assert rag_collection.count_documents({}) == 0

    async def test_a_missing_memory_raises(self, service):
        from bson import ObjectId

        with pytest.raises(MemoryNotFoundError):
            await service.delete_rag(str(ObjectId()))

    async def test_a_malformed_id_raises(self, service):
        with pytest.raises(MemoryNotFoundError):
            await service.delete_rag("nope")

    async def test_another_agent_cannot_delete_it(self, service, rag_collection):
        rag_id = seed(rag_collection, agent_id=AGENT_B)
        with pytest.raises(MemoryNotFoundError):
            await service.delete_rag(str(rag_id), agent_id=AGENT_A)
        assert rag_collection.count_documents({}) == 1

    async def test_a_wildcard_agent_cannot_delete_another_agents_memory(
        self, service, rag_collection
    ):
        rag_id = seed(rag_collection, agent_id=AGENT_B)
        with pytest.raises(MemoryNotFoundError):
            await service.delete_rag(str(rag_id), agent_id=".*")
        assert rag_collection.count_documents({}) == 1

    async def test_another_organization_cannot_delete_it(self, rag_collection):
        rag_id = seed(rag_collection)
        intruder = MongoVectorDbRagService(TenantCollection(rag_collection, OTHER_ORG))
        with pytest.raises(MemoryNotFoundError):
            await intruder.delete_rag(str(rag_id))
        assert rag_collection.count_documents({}) == 1


# --- embedding ------------------------------------------------------------


class TestUpdateEmbedding:
    async def test_the_enricher_is_invoked(self, service, rag_collection, monkeypatch):
        rag_id = seed(rag_collection)
        enricher = MagicMock()
        enricher.return_value.enrich_and_embed = AsyncMock()
        monkeypatch.setattr(
            "app.services.rag_enrichment_service.RagEnrichmentService", enricher
        )
        await service.update_embedding(str(rag_id))
        enricher.return_value.enrich_and_embed.assert_awaited_once_with(str(rag_id))

    async def test_a_failure_marks_the_memory_rather_than_raising(
        self, service, rag_collection, monkeypatch
    ):
        """This runs in a background pipeline, so raising would lose the
        record's state entirely; the UI needs to see that it errored."""
        rag_id = seed(rag_collection)
        enricher = MagicMock()
        enricher.return_value.enrich_and_embed = AsyncMock(
            side_effect=RuntimeError("embedding api down")
        )
        monkeypatch.setattr(
            "app.services.rag_enrichment_service.RagEnrichmentService", enricher
        )
        await service.update_embedding(str(rag_id))
        assert rag_collection.find_one({"_id": rag_id})["status"] == "error"

    async def test_a_malformed_id_does_not_raise(self, service, monkeypatch):
        enricher = MagicMock()
        enricher.return_value.enrich_and_embed = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        monkeypatch.setattr(
            "app.services.rag_enrichment_service.RagEnrichmentService", enricher
        )
        await service.update_embedding("not-an-id")


# --- search ---------------------------------------------------------------


class TestSearch:
    async def test_an_embedding_failure_yields_no_results(
        self, service, rag_collection, monkeypatch
    ):
        """Better an empty answer than a 500 on every chat turn."""
        seed(rag_collection, content="kita is a platform")
        monkeypatch.setattr(
            rag_module,
            "_create_embedding",
            AsyncMock(side_effect=RuntimeError("openrouter down")),
        )
        assert await service.search("kita") == []

    async def test_the_keyword_fallback_finds_a_match(
        self, service, rag_collection, no_embeddings
    ):
        """mongomock has no $vectorSearch, which is exactly the production
        case of an index that is missing or still building."""
        seed(rag_collection, title="Kita platform", content="an agent runtime")
        assert len(await service.search("platform")) == 1

    async def test_the_fallback_searches_the_content(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, title="Note", content="the deployment runbook")
        assert len(await service.search("runbook")) == 1

    async def test_the_fallback_searches_the_generated_question(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, title="Note", content="x", question="how do I deploy?")
        assert len(await service.search("deploy")) == 1

    async def test_stopwords_do_not_match_everything(
        self, service, rag_collection, no_embeddings
    ):
        """Without stripping them, "what is X" matches every memory
        containing "what"."""
        seed(rag_collection, title="Note", content="what a lovely day")
        assert await service.search("what about kita") == []

    async def test_a_query_of_only_stopwords_still_searches(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, title="Note", content="what")
        assert len(await service.search("what")) == 1

    async def test_regex_metacharacters_in_the_query_are_escaped(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, title="Note", content="plain text")
        assert await service.search("(.*)") == []

    async def test_no_matches_yields_an_empty_list(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, content="something else")
        assert await service.search("unrelated") == []

    async def test_the_limit_is_respected(self, service, rag_collection, no_embeddings):
        for i in range(5):
            seed(rag_collection, title=f"Note {i}", content="kita platform")
        assert len(await service.search("platform", limit=2)) == 2

    async def test_results_are_scoped_to_the_agent(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, agent_id=AGENT_B, content="kita platform")
        assert await service.search("platform", agent_id=AGENT_A) == []

    async def test_organization_wide_memories_are_searchable_by_any_agent(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, agent_id=None, content="kita platform")
        assert len(await service.search("platform", agent_id=AGENT_A)) == 1

    async def test_a_wildcard_agent_does_not_widen_the_search(
        self, service, rag_collection, no_embeddings
    ):
        seed(rag_collection, agent_id=AGENT_B, content="kita platform")
        assert await service.search("platform", agent_id=".*") == []

    async def test_another_organizations_memories_are_not_searchable(
        self, rag_collection, no_embeddings
    ):
        seed(rag_collection, content="kita platform")
        intruder = MongoVectorDbRagService(TenantCollection(rag_collection, OTHER_ORG))
        assert await intruder.search("platform") == []

    async def test_newer_memories_outrank_older_ones_at_equal_relevance(
        self, service, rag_collection, no_embeddings
    ):
        """The recency boost is what stops a year-old note from permanently
        outranking today's correction of it."""
        old = datetime.now(timezone.utc) - timedelta(days=400)
        seed(rag_collection, title="stale", content="kita platform", created_at=old)
        seed(rag_collection, title="fresh", content="kita platform")
        assert [r.title for r in await service.search("platform")][0] == "fresh"

    async def test_a_rerank_score_outweighs_recency(
        self, service, rag_collection, monkeypatch
    ):
        old = datetime.now(timezone.utc) - timedelta(days=400)
        seed(rag_collection, title="stale", content="kita platform", created_at=old)
        seed(rag_collection, title="fresh", content="kita platform")
        monkeypatch.setattr(
            rag_module, "_create_embedding", AsyncMock(return_value=[0.1] * 8)
        )

        async def rerank(query, documents, top_n):
            # Score whichever candidate is the stale one far higher.
            return [
                {"index": i, "relevance_score": 9.0 if "stale" in d else 0.0}
                for i, d in enumerate(documents)
            ]

        monkeypatch.setattr(rag_module, "_rerank", rerank)
        assert [r.title for r in await service.search("platform")][0] == "stale"

    async def test_a_rerank_failure_falls_back_to_the_local_ordering(
        self, service, rag_collection, monkeypatch
    ):
        seed(rag_collection, content="kita platform")
        monkeypatch.setattr(
            rag_module, "_create_embedding", AsyncMock(return_value=[0.1] * 8)
        )
        monkeypatch.setattr(
            rag_module, "_rerank", AsyncMock(side_effect=RuntimeError("rerank down"))
        )
        assert len(await service.search("platform")) == 1

    async def test_a_string_timestamp_is_handled(
        self, service, rag_collection, no_embeddings
    ):
        """Documents written by other services can carry ISO strings rather
        than BSON dates."""
        seed(
            rag_collection,
            content="kita platform",
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        assert len(await service.search("platform")) == 1

    async def test_one_corrupt_timestamp_does_not_take_down_the_search(
        self, service, rag_collection, no_embeddings
    ):
        """The scoring loop already tolerates an unparseable date, but the
        same value then raised out of RagResponse -- so a single malformed
        row failed the whole listing instead of just itself."""
        seed(rag_collection, content="kita platform", created_at="not a date")
        seed(rag_collection, content="kita platform")
        assert len(await service.search("platform")) == 2


# --- response shape -------------------------------------------------------


class TestFormatRagResponse:
    def test_the_id_is_stringified(self, rag_collection):
        rag_id = seed(rag_collection)
        doc = rag_collection.find_one({"_id": rag_id})
        assert format_rag_response(doc).id == str(rag_id)

    def test_the_status_defaults_to_pending(self, rag_collection):
        rag_id = seed(rag_collection)
        doc = rag_collection.find_one({"_id": rag_id})
        doc.pop("status")
        assert format_rag_response(doc).status == "pending"

    def test_the_enriched_fields_are_optional(self, rag_collection):
        rag_id = seed(rag_collection)
        doc = rag_collection.find_one({"_id": rag_id})
        response = format_rag_response(doc)
        assert response.question is None and response.answer is None
