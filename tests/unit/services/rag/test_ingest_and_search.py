"""Tests for the knowledge-base ingest and search services.

`IngestService` turns a parsed document into embedded leaves; the vector and
text search services read them back. The agent filter these share is the one
that decides which agent can see which leaf, and it reaches Mongo as a regex
built from a caller-supplied id -- so the escaping tests carry the same weight
here as they do in rag_service.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db import TenantCollection
from app.models.rag import RagCreateRequest, RagUpdateRequest
from app.services.rag.ingest_service import IngestService, format_rag_response
from app.services.rag.mongodb_text_search_rag_service import MongoDBTextSearchRagService
from app.services.rag.reranking_rag_service import RerankingRagService

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"
AGENT_A = "6a6700000000000000000001"
AGENT_B = "6a6700000000000000000002"


@pytest.fixture
def leaves_collection(mongo_db):
    return mongo_db["file_parsed_flattened"]


@pytest.fixture
def parse_collection(mongo_db):
    return mongo_db["file_parse"]


@pytest.fixture
def vector_service() -> MagicMock:
    service = MagicMock(name="vector_service")
    service.create_embedding = AsyncMock(return_value=[0.1] * 8)
    service.bulk_create_embeddings = AsyncMock(return_value=[[0.1] * 8] * 10)
    return service


@pytest.fixture
def enrichment_service() -> MagicMock:
    service = MagicMock(name="nested_data_enrichment_service")
    service.build_hierarchy_and_leaves.return_value = ({}, [])
    return service


@pytest.fixture
def service(
    leaves_collection, parse_collection, vector_service, enrichment_service
) -> IngestService:
    return IngestService(
        collection=TenantCollection(leaves_collection, ORG_ID),
        vector_service=vector_service,
        nested_data_enrichment_service=enrichment_service,
        parse_collection=TenantCollection(parse_collection, ORG_ID),
    )


def a_leaf(**overrides) -> dict:
    leaf = {
        "file_id": "file_1",
        "text": "some text",
        "heading_to_text": "Heading > some text",
        "content": "some text",
        "json_path": "$.a",
        "breadcrumb": "a",
        "heading_text": "Heading",
        "page": 1,
    }
    leaf.update(overrides)
    return leaf


# --- manual entries -------------------------------------------------------


class TestAddRag:
    async def test_an_entry_is_created(self, service):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert created.title == "T"

    async def test_it_starts_pending(self, service):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert created.status == "pending"

    async def test_the_searchable_text_combines_title_and_content(
        self, service, leaves_collection
    ):
        """Search runs against `text`, not the fields separately, so a title
        that never made it in is a title that can never be found."""
        await service.add_rag(RagCreateRequest(title="Runbook", content="restart it"))
        stored = leaves_collection.find_one({})
        assert stored["text"] == "Runbook: restart it"
        assert stored["heading_to_text"] == "Runbook > restart it"

    async def test_it_is_scoped_to_the_organization(self, service, leaves_collection):
        await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert leaves_collection.find_one({})["org_id"] == ORG_ID

    async def test_a_versioned_agent_id_is_stored_as_its_base(
        self, service, leaves_collection
    ):
        await service.add_rag(
            RagCreateRequest(title="T", content="C", agent_id=f"{AGENT_A}-v2")
        )
        assert leaves_collection.find_one({})["agent_id"] == AGENT_A

    async def test_a_manual_entry_is_marked_as_such(self, service, leaves_collection):
        """These sit alongside document-derived leaves; the marker is how a
        re-ingest of the source file avoids deleting hand-written notes."""
        await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert leaves_collection.find_one({})["json_path"] == "manual_input"


class TestEditRag:
    async def test_renaming_an_entry_is_visible(self, service):
        """Every read path resolves the title through `heading_text`, so
        writing only `title` returned 200 and changed nothing the user
        could see."""
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        updated = await service.edit_rag(created.id, RagUpdateRequest(title="New"))
        assert updated.title == "New"

    async def test_renaming_an_entry_reaches_the_search_text(
        self, service, leaves_collection
    ):
        """The title is half of what search matches on; leaving it stale made
        an entry findable only by its old name."""
        created = await service.add_rag(RagCreateRequest(title="Old", content="body"))
        await service.edit_rag(created.id, RagUpdateRequest(title="New"))
        assert leaves_collection.find_one({})["text"] == "New: body"

    async def test_renaming_requeues_the_embedding(self, service):
        """The vector was built from the old text, so it no longer matches
        what the entry says."""
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        updated = await service.edit_rag(created.id, RagUpdateRequest(title="New"))
        assert updated.status == "pending"

    async def test_editing_the_content_keeps_the_existing_title(self, service):
        created = await service.add_rag(RagCreateRequest(title="Keep", content="C"))
        updated = await service.edit_rag(created.id, RagUpdateRequest(content="new"))
        assert updated.title == "Keep"

    async def test_editing_the_content_requeues_the_embedding(self, service):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        updated = await service.edit_rag(created.id, RagUpdateRequest(content="new"))
        assert updated.status == "pending"

    async def test_editing_the_content_rebuilds_the_searchable_text(
        self, service, leaves_collection
    ):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        await service.edit_rag(created.id, RagUpdateRequest(content="new body"))
        assert leaves_collection.find_one({})["text"].endswith("new body")

    async def test_an_empty_update_is_a_no_op(self, service):
        created = await service.add_rag(RagCreateRequest(title="Original", content="C"))
        assert (await service.edit_rag(created.id, RagUpdateRequest())).title == (
            "Original"
        )

    async def test_a_malformed_id_is_rejected(self, service):
        with pytest.raises(ValueError):
            await service.edit_rag("nope", RagUpdateRequest(title="x"))


class TestDeleteRag:
    async def test_an_entry_is_deleted(self, service, leaves_collection):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert await service.delete_rag(created.id) is True
        assert leaves_collection.count_documents({}) == 0

    async def test_deleting_a_missing_entry_reports_failure(self, service):
        from bson import ObjectId

        assert await service.delete_rag(str(ObjectId())) is False

    async def test_a_malformed_id_is_rejected(self, service):
        with pytest.raises(ValueError):
            await service.delete_rag("nope")

    async def test_another_organization_cannot_delete_it(
        self, service, leaves_collection, vector_service, enrichment_service
    ):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        intruder = IngestService(
            collection=TenantCollection(leaves_collection, OTHER_ORG),
            vector_service=vector_service,
            nested_data_enrichment_service=enrichment_service,
        )
        assert await intruder.delete_rag(created.id) is False
        assert leaves_collection.count_documents({}) == 1


class TestReads:
    async def test_an_entry_is_returned(self, service):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        assert service.get_rag(created.id).id == created.id

    async def test_a_missing_entry_yields_nothing(self, service):
        from bson import ObjectId

        assert service.get_rag(str(ObjectId())) is None

    async def test_a_malformed_id_is_rejected(self, service):
        with pytest.raises(ValueError):
            service.get_rag("nope")

    async def test_entries_are_listed_newest_first(self, service, leaves_collection):
        old = datetime.now(timezone.utc) - timedelta(days=2)
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, heading_text="old", created_at=old, updated_at=old)
        )
        await service.add_rag(RagCreateRequest(title="new", content="C"))
        assert [r.title for r in service.get_all_rags()][0] == "new"

    async def test_another_organizations_entries_are_not_listed(
        self, service, leaves_collection, vector_service, enrichment_service
    ):
        await service.add_rag(RagCreateRequest(title="T", content="C"))
        intruder = IngestService(
            collection=TenantCollection(leaves_collection, OTHER_ORG),
            vector_service=vector_service,
            nested_data_enrichment_service=enrichment_service,
        )
        assert intruder.get_all_rags() == []


# --- the agent filter -----------------------------------------------------


class TestAgentFilter:
    def _scoped(self, leaves_collection, vector_service, enrichment_service, agent_id):
        return IngestService(
            collection=TenantCollection(leaves_collection, ORG_ID),
            vector_service=vector_service,
            nested_data_enrichment_service=enrichment_service,
            agent_id=agent_id,
        )

    def test_no_agent_means_no_filter(self, service):
        assert service._get_agent_filter() == {}

    @pytest.mark.parametrize("hostile", [".*", "^.*$", "a|b", "(a+)+"])
    def test_regex_metacharacters_are_treated_as_literals(
        self, leaves_collection, vector_service, enrichment_service, hostile
    ):
        """This filter also guards edit and delete, so an id matching every
        agent would let a caller rewrite another agent's knowledge base."""
        leaves_collection.insert_one(a_leaf(org_id=ORG_ID, agent_id=AGENT_A))
        scoped = self._scoped(
            leaves_collection, vector_service, enrichment_service, hostile
        )
        assert scoped.get_all_rags() == []

    def test_an_agent_sees_its_own_and_the_shared_entries(
        self, leaves_collection, vector_service, enrichment_service
    ):
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, agent_id=AGENT_A, heading_text="mine")
        )
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, agent_id=None, heading_text="shared")
        )
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, agent_id=AGENT_B, heading_text="theirs")
        )
        scoped = self._scoped(
            leaves_collection, vector_service, enrichment_service, AGENT_A
        )
        assert sorted(r.title for r in scoped.get_all_rags()) == ["mine", "shared"]

    def test_a_versioned_id_matches_its_base(
        self, leaves_collection, vector_service, enrichment_service
    ):
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, agent_id=AGENT_A, heading_text="mine")
        )
        scoped = self._scoped(
            leaves_collection, vector_service, enrichment_service, f"{AGENT_A}-v7"
        )
        assert [r.title for r in scoped.get_all_rags()] == ["mine"]


# --- embedding ------------------------------------------------------------


class TestUpdateEmbedding:
    async def test_the_entry_becomes_searchable(
        self, service, leaves_collection, vector_service
    ):
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        await service.update_embedding(created.id)
        stored = leaves_collection.find_one({})
        assert stored["status"] == "completed" and stored["embedding"]

    async def test_the_heading_context_is_what_gets_embedded(
        self, service, vector_service
    ):
        """Embedding the bare text loses the heading trail, which is most of
        what disambiguates one row of a table from another."""
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        await service.update_embedding(created.id)
        assert vector_service.create_embedding.await_args[0][0] == "T > C"

    async def test_a_missing_entry_is_a_no_op(self, service, vector_service):
        from bson import ObjectId

        await service.update_embedding(str(ObjectId()))
        vector_service.create_embedding.assert_not_awaited()

    async def test_a_failure_marks_the_entry(
        self, service, leaves_collection, vector_service
    ):
        """This runs in a background pipeline; raising would lose the state
        and leave the row pending forever."""
        created = await service.add_rag(RagCreateRequest(title="T", content="C"))
        vector_service.create_embedding = AsyncMock(side_effect=RuntimeError("down"))
        await service.update_embedding(created.id)
        assert leaves_collection.find_one({})["status"] == "error"

    async def test_a_malformed_id_does_not_raise(self, service):
        await service.update_embedding("not-an-id")


# --- document ingestion ---------------------------------------------------


class TestIngestFileParse:
    async def test_without_a_parse_collection_nothing_is_ingested(
        self, leaves_collection, vector_service, enrichment_service
    ):
        service = IngestService(
            collection=TenantCollection(leaves_collection, ORG_ID),
            vector_service=vector_service,
            nested_data_enrichment_service=enrichment_service,
        )
        assert await service.ingest_file_parse("file_1", ORG_ID) is False

    async def test_an_unparsed_file_is_not_ingested(self, service):
        assert await service.ingest_file_parse("file_missing", ORG_ID) is False

    async def test_a_parse_with_no_leaves_is_not_ingested(
        self, service, parse_collection
    ):
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {}}
        )
        assert await service.ingest_file_parse("file_1", ORG_ID) is False

    async def test_leaves_are_stored(
        self, service, parse_collection, leaves_collection, enrichment_service
    ):
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        )
        enrichment_service.build_hierarchy_and_leaves.return_value = (
            {},
            [a_leaf(), a_leaf()],
        )
        assert await service.ingest_file_parse("file_1", ORG_ID) is True
        assert leaves_collection.count_documents({}) == 2

    async def test_the_leaves_are_embedded_in_one_call(
        self, service, parse_collection, enrichment_service, vector_service
    ):
        """One request per leaf would mean hundreds of round trips for a
        single document."""
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        )
        enrichment_service.build_hierarchy_and_leaves.return_value = (
            {},
            [a_leaf(), a_leaf()],
        )
        await service.ingest_file_parse("file_1", ORG_ID)
        vector_service.bulk_create_embeddings.assert_awaited_once()

    async def test_the_leaves_point_back_at_the_parse_record(
        self, service, parse_collection, leaves_collection, enrichment_service
    ):
        parse_id = parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        ).inserted_id
        enrichment_service.build_hierarchy_and_leaves.return_value = ({}, [a_leaf()])
        await service.ingest_file_parse("file_1", ORG_ID)
        assert leaves_collection.find_one({})["parent_doc_id"] == parse_id

    async def test_re_ingesting_replaces_rather_than_duplicates(
        self, service, parse_collection, leaves_collection, enrichment_service
    ):
        """A re-parse that appended would double every answer's context."""
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        )
        enrichment_service.build_hierarchy_and_leaves.return_value = ({}, [a_leaf()])
        await service.ingest_file_parse("file_1", ORG_ID)
        await service.ingest_file_parse("file_1", ORG_ID)
        assert leaves_collection.count_documents({}) == 1

    async def test_re_ingesting_leaves_other_files_alone(
        self, service, parse_collection, leaves_collection, enrichment_service
    ):
        leaves_collection.insert_one(a_leaf(org_id=ORG_ID, file_id="other_file"))
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        )
        enrichment_service.build_hierarchy_and_leaves.return_value = ({}, [a_leaf()])
        await service.ingest_file_parse("file_1", ORG_ID)
        assert leaves_collection.count_documents({"file_id": "other_file"}) == 1

    async def test_more_leaves_than_embeddings_is_survivable(
        self,
        service,
        parse_collection,
        leaves_collection,
        enrichment_service,
        vector_service,
    ):
        """A truncated embedding response should cost those leaves their
        vector, not fail the whole document."""
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        )
        enrichment_service.build_hierarchy_and_leaves.return_value = (
            {},
            [a_leaf(), a_leaf()],
        )
        vector_service.bulk_create_embeddings = AsyncMock(return_value=[[0.1] * 8])
        assert await service.ingest_file_parse("file_1", ORG_ID) is True
        assert leaves_collection.count_documents({"embedding": None}) == 1

    async def test_another_organizations_parse_is_not_visible(
        self, service, parse_collection
    ):
        parse_collection.insert_one(
            {"org_id": OTHER_ORG, "file_id": "file_1", "result": {"a": 1}}
        )
        assert await service.ingest_file_parse("file_1", ORG_ID) is False


# --- text search ----------------------------------------------------------


@pytest.fixture
def text_service(leaves_collection) -> MongoDBTextSearchRagService:
    return MongoDBTextSearchRagService(TenantCollection(leaves_collection, ORG_ID))


class TestTextSearch:
    async def test_a_match_is_found(self, text_service, leaves_collection):
        """mongomock has no text index, so this exercises the regex fallback
        -- the same path production takes before the index is built."""
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, text="the deployment runbook")
        )
        assert len(await text_service.text_search("runbook")) == 1

    async def test_no_match_yields_nothing(self, text_service, leaves_collection):
        leaves_collection.insert_one(a_leaf(org_id=ORG_ID, text="unrelated"))
        assert await text_service.text_search("runbook") == []

    async def test_stopwords_do_not_match_everything(
        self, text_service, leaves_collection
    ):
        leaves_collection.insert_one(a_leaf(org_id=ORG_ID, text="what a nice day"))
        assert await text_service.text_search("what about kita") == []

    async def test_a_query_of_only_stopwords_still_searches(
        self, text_service, leaves_collection
    ):
        leaves_collection.insert_one(a_leaf(org_id=ORG_ID, text="what"))
        assert len(await text_service.text_search("what")) == 1

    async def test_regex_metacharacters_in_the_query_are_escaped(
        self, text_service, leaves_collection
    ):
        leaves_collection.insert_one(a_leaf(org_id=ORG_ID, text="plain text"))
        assert await text_service.text_search("(.*)") == []

    async def test_the_limit_is_respected(self, text_service, leaves_collection):
        for _ in range(5):
            leaves_collection.insert_one(a_leaf(org_id=ORG_ID, text="runbook"))
        assert len(await text_service.text_search("runbook", limit=2)) == 2

    async def test_results_are_scoped_to_the_agent(
        self, text_service, leaves_collection
    ):
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, agent_id=AGENT_B, text="runbook")
        )
        assert await text_service.text_search("runbook", agent_id=AGENT_A) == []

    async def test_a_wildcard_agent_does_not_widen_the_search(
        self, text_service, leaves_collection
    ):
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, agent_id=AGENT_B, text="runbook")
        )
        assert await text_service.text_search("runbook", agent_id=".*") == []

    async def test_shared_entries_are_searchable_by_any_agent(
        self, text_service, leaves_collection
    ):
        leaves_collection.insert_one(
            a_leaf(org_id=ORG_ID, agent_id=None, text="runbook")
        )
        assert len(await text_service.text_search("runbook", agent_id=AGENT_A)) == 1

    async def test_another_organizations_entries_are_not_searchable(
        self, leaves_collection
    ):
        leaves_collection.insert_one(a_leaf(org_id=ORG_ID, text="runbook"))
        intruder = MongoDBTextSearchRagService(
            TenantCollection(leaves_collection, OTHER_ORG)
        )
        assert await intruder.text_search("runbook") == []


# --- reranking ------------------------------------------------------------


class TestReranking:
    @pytest.fixture
    def service(self) -> RerankingRagService:
        return RerankingRagService()

    async def test_no_candidates_yields_nothing(self, service):
        assert await service.rerank("q", []) == []

    async def test_an_api_failure_preserves_the_original_order(
        self, service, monkeypatch
    ):
        """Retrieval already ranked these; falling back to that beats
        returning nothing because a third party is down."""
        candidates = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        assert await service.rerank("q", candidates, limit=2) == candidates[:2]

    async def test_the_scores_reorder_the_candidates(self, service, monkeypatch):
        candidates = [{"text": "a"}, {"text": "b"}]

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "results": [
                        {"index": 0, "relevance_score": 0.1},
                        {"index": 1, "relevance_score": 0.9},
                    ]
                }

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *args, **kwargs):
                return Response()

        monkeypatch.setattr("httpx.AsyncClient", lambda **kw: Client())
        assert [d["text"] for d in await service.rerank("q", candidates)] == ["b", "a"]

    async def test_an_out_of_range_index_is_ignored(self, service, monkeypatch):
        """A reranker echoing an index we never sent would otherwise raise
        mid-response."""
        candidates = [{"text": "a"}]

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [{"index": 99, "relevance_score": 0.9}]}

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *args, **kwargs):
                return Response()

        monkeypatch.setattr("httpx.AsyncClient", lambda **kw: Client())
        assert await service.rerank("q", candidates) == []


# --- response shape -------------------------------------------------------


class TestFormatRagResponse:
    def test_the_id_is_stringified(self, leaves_collection):
        leaf_id = leaves_collection.insert_one(
            a_leaf(
                org_id=ORG_ID,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ).inserted_id
        doc = leaves_collection.find_one({"_id": leaf_id})
        assert format_rag_response(doc).id == str(leaf_id)
