"""Tests for app.services.rag.retrieval_service.

The pipeline is: search vector and text in parallel, fuse the two rankings
with RRF, auto-merge sibling leaves back up to their parent when enough of
them hit, resolve each candidate's sub-tree, then rerank twice. The fusion and
merge steps are pure functions over the candidate list, which is where the
behaviour actually lives -- everything either side is a call to a service that
is mocked here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.db import TenantCollection
from app.services.rag.retrieval_service import RetrievalService

ORG_ID = "org_test_0001"


@pytest.fixture
def leaves_collection(mongo_db):
    return mongo_db["file_parsed_flattened"]


@pytest.fixture
def parse_collection(mongo_db):
    return mongo_db["file_parse"]


@pytest.fixture
def vector_service() -> MagicMock:
    service = MagicMock(name="vector_service")
    service.vector_search = AsyncMock(return_value=[])
    return service


@pytest.fixture
def text_service() -> MagicMock:
    service = MagicMock(name="text_service")
    service.text_search = AsyncMock(return_value=[])
    return service


@pytest.fixture
def reranking_service() -> MagicMock:
    service = MagicMock(name="reranking_service")
    # Identity rerank: keep the order, honour the limit.
    service.rerank = AsyncMock(
        side_effect=lambda query, candidates, limit=5: candidates[:limit]
    )
    return service


@pytest.fixture
def enrichment_service() -> MagicMock:
    service = MagicMock(name="nested_data_enrichment_service")
    service.build_hierarchy_and_leaves.return_value = ({}, [])
    return service


@pytest.fixture
def service(
    leaves_collection,
    parse_collection,
    vector_service,
    text_service,
    reranking_service,
    enrichment_service,
) -> RetrievalService:
    return RetrievalService(
        collection=TenantCollection(leaves_collection, ORG_ID),
        vector_service=vector_service,
        text_service=text_service,
        reranking_service=reranking_service,
        parse_collection=TenantCollection(parse_collection, ORG_ID),
        nested_data_enrichment_service=enrichment_service,
    )


def a_candidate(**overrides) -> dict:
    doc = {
        "_id": ObjectId(),
        "org_id": ORG_ID,
        "file_id": "file_1",
        "json_path": "manual_input",
        "heading_text": "Heading",
        "text": "some text",
        "heading_to_text": "Heading > some text",
        "content": "some text",
        "page": 1,
    }
    doc.update(overrides)
    return doc


# --- rank fusion ----------------------------------------------------------


class TestReciprocalRankFusion:
    def test_two_empty_lists_fuse_to_nothing(self, service):
        assert service._apply_rrf([], []) == []

    def test_one_list_passes_through_in_order(self, service):
        docs = [a_candidate(), a_candidate(), a_candidate()]
        assert service._apply_rrf(docs, []) == docs

    def test_a_document_in_both_lists_outranks_one_in_either(self, service):
        """That is the point of fusion: agreement between two independent
        retrievers is stronger evidence than a high rank in one."""
        both = a_candidate(heading_text="both")
        vector_only = a_candidate(heading_text="vector only")
        text_only = a_candidate(heading_text="text only")
        fused = service._apply_rrf([vector_only, both], [text_only, both])
        assert fused[0]["heading_text"] == "both"

    def test_duplicates_are_collapsed(self, service):
        doc = a_candidate()
        assert len(service._apply_rrf([doc], [doc])) == 1

    def test_a_higher_rank_scores_higher(self, service):
        first = a_candidate(heading_text="first")
        second = a_candidate(heading_text="second")
        fused = service._apply_rrf([first, second], [])
        assert [d["heading_text"] for d in fused] == ["first", "second"]

    def test_the_limit_is_respected(self, service):
        docs = [a_candidate() for _ in range(10)]
        assert len(service._apply_rrf(docs, [], limit=3)) == 3


# --- sibling auto-merging -------------------------------------------------


class TestAutoMergeSiblings:
    def test_nothing_to_merge_passes_through(self, service):
        docs = [a_candidate(json_path="a.b")]
        assert service._auto_merge_siblings(docs, {}, 0.35) == docs

    def test_a_top_level_path_is_never_merged(self, service):
        """There is no parent to merge into."""
        docs = [a_candidate(json_path="a")]
        assert service._auto_merge_siblings(docs, {}, 0.35) == docs

    def test_a_candidate_without_a_file_is_kept_as_is(self, service):
        docs = [a_candidate(file_id=None, json_path="a.b")]
        assert service._auto_merge_siblings(docs, {}, 0.35) == docs

    def test_enough_sibling_hits_merge_into_the_parent(self, service):
        """Returning three sibling rows separately gives the model three
        fragments of one table; the parent is the coherent unit."""
        trees = {"file_1": {"section": {"a": 1, "b": 2, "c": 3, "d": 4}}}
        docs = [
            a_candidate(json_path="section.a", heading_text="Section > A"),
            a_candidate(json_path="section.b", heading_text="Section > B"),
        ]
        merged = service._auto_merge_siblings(docs, trees, 0.35)
        assert len(merged) == 1 and merged[0]["json_path"] == "section"

    def test_too_few_sibling_hits_stay_separate(self, service):
        trees = {"file_1": {"section": {f"k{i}": i for i in range(10)}}}
        docs = [a_candidate(json_path="section.k0")]
        merged = service._auto_merge_siblings(docs, trees, 0.35)
        assert merged[0]["json_path"] == "section.k0"

    def test_the_merged_candidate_drops_the_leaf_from_its_heading(self, service):
        trees = {"file_1": {"section": {"a": 1, "b": 2}}}
        docs = [
            a_candidate(json_path="section.a", heading_text="Doc > Section > A"),
            a_candidate(json_path="section.b", heading_text="Doc > Section > B"),
        ]
        merged = service._auto_merge_siblings(docs, trees, 0.35)
        assert merged[0]["heading_text"] == "Doc > Section"

    def test_an_unknown_file_is_left_alone(self, service):
        docs = [a_candidate(json_path="a.b")]
        assert service._auto_merge_siblings(docs, {"other": {}}, 0.35) == docs

    def test_a_non_dict_parent_is_not_merged(self, service):
        """A list or scalar parent has no sibling keys to count."""
        trees = {"file_1": {"section": "just a string"}}
        docs = [a_candidate(json_path="section.a")]
        assert service._auto_merge_siblings(docs, trees, 0.35)[0]["json_path"] == (
            "section.a"
        )

    def test_unmerged_candidates_survive_alongside_merged_ones(self, service):
        trees = {"file_1": {"section": {"a": 1, "b": 2}, "other": "x"}}
        docs = [
            a_candidate(json_path="section.a"),
            a_candidate(json_path="section.b"),
            a_candidate(json_path="elsewhere.z"),
        ]
        merged = service._auto_merge_siblings(docs, trees, 0.35)
        assert sorted(d["json_path"] for d in merged) == ["elsewhere.z", "section"]


# --- sub-tree serialisation -----------------------------------------------


class TestSerializeSubTree:
    def test_a_scalar_is_stringified(self, service):
        assert service._serialize_sub_tree("", 42) == "42"

    def test_a_heading_is_prefixed(self, service):
        """The heading trail is most of what tells the model which section of
        the document a value came from."""
        assert service._serialize_sub_tree("Doc > Section", 42).startswith(
            "Location: Doc > Section"
        )

    def test_a_flat_mapping_becomes_lines(self, service):
        assert service._serialize_sub_tree("", {"a": 1, "b": 2}) == "a: 1\nb: 2"

    def test_a_nested_mapping_is_json_encoded(self, service):
        result = service._serialize_sub_tree("", {"a": {"b": 1}})
        assert result == 'a: {"b": 1}'

    def test_an_empty_mapping_yields_nothing(self, service):
        assert service._serialize_sub_tree("", {}) == ""


# --- the full pipeline ----------------------------------------------------


class TestSearch:
    async def test_no_hits_yields_nothing(self, service):
        assert await service.search("q") == []

    async def test_both_retrievers_are_queried(
        self, service, vector_service, text_service
    ):
        """Running them in sequence would double the latency of every turn."""
        await service.search("q")
        vector_service.vector_search.assert_awaited_once()
        text_service.text_search.assert_awaited_once()

    async def test_a_manual_entry_is_returned(self, service, vector_service):
        vector_service.vector_search = AsyncMock(
            return_value=[a_candidate(json_path="manual_input")]
        )
        results = await service.search("q")
        assert len(results) == 1 and results[0].content == "Heading > some text"

    async def test_a_document_leaf_resolves_its_sub_tree(
        self, service, vector_service, parse_collection, enrichment_service
    ):
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": {"b": "value"}}}
        )
        enrichment_service.build_hierarchy_and_leaves.return_value = (
            {"a": {"b": "value"}},
            [],
        )
        vector_service.vector_search = AsyncMock(
            return_value=[a_candidate(json_path="a.b", heading_text="A > B")]
        )
        results = await service.search("q")
        assert "value" in results[0].content

    async def test_a_leaf_whose_path_no_longer_exists_is_dropped(
        self, service, vector_service, parse_collection, enrichment_service
    ):
        """The document was re-parsed into a different shape; returning the
        stale leaf would quote text that is no longer in the file."""
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        )
        enrichment_service.build_hierarchy_and_leaves.return_value = ({"a": 1}, [])
        vector_service.vector_search = AsyncMock(
            return_value=[a_candidate(json_path="gone.missing")]
        )
        assert await service.search("q") == []

    async def test_duplicates_from_the_two_retrievers_are_returned_once(
        self, service, vector_service, text_service
    ):
        doc = a_candidate()
        vector_service.vector_search = AsyncMock(return_value=[doc])
        text_service.text_search = AsyncMock(return_value=[doc])
        assert len(await service.search("q")) == 1

    async def test_the_limit_is_respected(self, service, vector_service):
        vector_service.vector_search = AsyncMock(
            return_value=[a_candidate() for _ in range(10)]
        )
        assert len(await service.search("q", limit=3)) == 3

    async def test_the_reranker_sees_the_resolved_context(
        self, service, vector_service, reranking_service
    ):
        """Reranking the raw leaf text rather than the resolved sub-tree would
        score a fragment instead of the passage actually returned."""
        vector_service.vector_search = AsyncMock(return_value=[a_candidate()])
        await service.search("q")
        candidates = reranking_service.rerank.await_args_list[0][0][1]
        assert candidates[0]["context_text"]

    async def test_the_query_becomes_the_response_question(
        self, service, vector_service
    ):
        vector_service.vector_search = AsyncMock(return_value=[a_candidate()])
        assert (await service.search("what is kita"))[0].question == "what is kita"

    async def test_the_agent_scope_reaches_both_retrievers(
        self,
        leaves_collection,
        parse_collection,
        vector_service,
        text_service,
        reranking_service,
        enrichment_service,
    ):
        scoped = RetrievalService(
            collection=TenantCollection(leaves_collection, ORG_ID),
            vector_service=vector_service,
            text_service=text_service,
            reranking_service=reranking_service,
            parse_collection=TenantCollection(parse_collection, ORG_ID),
            nested_data_enrichment_service=enrichment_service,
            agent_id="agent_1",
        )
        await scoped.search("q")
        assert vector_service.vector_search.await_args.kwargs["agent_id"] == "agent_1"
        assert text_service.text_search.await_args.kwargs["agent_id"] == "agent_1"

    async def test_only_the_matched_files_parse_records_are_loaded(
        self, service, vector_service, parse_collection, enrichment_service
    ):
        """Rebuilding every document's tree on every search would put the
        whole knowledge base through the enricher per query."""
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_1", "result": {"a": 1}}
        )
        parse_collection.insert_one(
            {"org_id": ORG_ID, "file_id": "file_2", "result": {"b": 2}}
        )
        vector_service.vector_search = AsyncMock(
            return_value=[a_candidate(file_id="file_1", json_path="a")]
        )
        await service.search("q")
        rebuilt = {
            call.kwargs["file_id"]
            for call in enrichment_service.build_hierarchy_and_leaves.call_args_list
        }
        assert rebuilt == {"file_1"}
