"""Tests for app.services.graph_rag_service.

Everything here runs against Neo4j, so the driver is stubbed and what gets
asserted is the Cypher and parameters the service sends. Two things matter:
every statement carries org_id (Neo4j has no equivalent of TenantCollection,
so the isolation is hand-written into each query), and labels and
relationship types are validated, because Cypher cannot parameterize those
and they end up in the query string.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.graph_rag import (
    GraphChunk,
    GraphDocument,
    GraphEntity,
    GraphRelationship,
)
from app.services.graph_rag_service import Neo4JGraphRagService

ORG_ID = "org_test_0001"


class FakeSession:
    """Records every statement the service runs."""

    def __init__(self, single_result=None):
        self.statements: list[tuple[str, dict]] = []
        self._single = single_result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.statements.append((query, params))
        result = MagicMock()
        result.single = AsyncMock(return_value=self._single)
        result.__aiter__ = lambda _self: iter([])
        return result

    @property
    def queries(self) -> str:
        return "\n".join(q for q, _ in self.statements)

    @property
    def params(self) -> list[dict]:
        return [p for _, p in self.statements]


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def service(monkeypatch, session) -> Neo4JGraphRagService:
    driver = MagicMock(name="driver")
    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()
    monkeypatch.setattr(
        "neo4j.AsyncGraphDatabase.driver", MagicMock(return_value=driver)
    )
    built = Neo4JGraphRagService("bolt://x", "neo4j", "pw", ORG_ID)
    built.driver = driver
    # The embedding model is a 90MB download; stub it.
    built._model = MagicMock(
        encode=MagicMock(return_value=MagicMock(tolist=lambda: [0.1] * 8))
    )
    return built


def a_document(**overrides) -> GraphDocument:
    payload = {"id": "doc_1", "title": "Handbook", "metadata": {"source": "upload"}}
    payload.update(overrides)
    return GraphDocument(**payload)


def a_chunk(**overrides) -> GraphChunk:
    payload = {
        "id": "chunk_1",
        "document_id": "doc_1",
        "content": "free coffee",
        "metadata": {"agent_id": "agent_1", "filename": "handbook.pdf"},
        "embedding": [0.1] * 8,
    }
    payload.update(overrides)
    return GraphChunk(**payload)


# --- identifier validation ------------------------------------------------


class TestIdentifierValidation:
    @pytest.mark.parametrize("label", ["Entity", "Concept", "_Private", "A1"])
    def test_plain_identifiers_are_accepted(self, service, label):
        assert service._safe_identifier(label, "node label") == label

    @pytest.mark.parametrize(
        "hostile",
        [
            "Entity) DETACH DELETE (n",
            "Entity {x: 1}",
            "Entity-Name",
            "1Entity",
            "",
            None,
        ],
    )
    def test_anything_executable_is_rejected(self, service, hostile):
        """Cypher cannot parameterize a label, so it is interpolated into the
        query string -- an unvalidated one is executable, not just wrong."""
        with pytest.raises(ValueError):
            service._safe_identifier(hostile, "node label")

    async def test_a_hostile_label_never_reaches_the_database(self, service, session):
        with pytest.raises(ValueError):
            await service.upsert_node("X) DETACH DELETE (n", {"id": "1"})
        assert session.statements == []

    async def test_a_hostile_relationship_type_never_reaches_the_database(
        self, service, session
    ):
        with pytest.raises(ValueError):
            await service.upsert_relationship("a", "b", "R]->() DETACH DELETE (n", {})
        assert session.statements == []


# --- property preparation -------------------------------------------------


class TestPrepareProps:
    def test_scalars_pass_through(self, service):
        assert service._prepare_props({"a": 1, "b": "x", "c": True}) == {
            "a": 1,
            "b": "x",
            "c": True,
        }

    def test_a_mapping_is_serialised(self, service):
        """Neo4j stores primitives and flat arrays of them; a map property is
        rejected by the driver. The old condition iterated an empty list for a
        dict, so `all(...)` was vacuously true and no mapping was serialised."""
        assert service._prepare_props({"a": {"b": 1}})["a"] == '{"b": 1}'

    def test_a_nested_mapping_is_serialised(self, service):
        assert isinstance(service._prepare_props({"a": {"b": {"c": 1}}})["a"], str)

    def test_a_flat_list_of_primitives_is_kept(self, service):
        assert service._prepare_props({"a": [1, 2, 3]})["a"] == [1, 2, 3]

    def test_a_list_of_mappings_is_serialised(self, service):
        assert isinstance(service._prepare_props({"a": [{"b": 1}]})["a"], str)

    def test_an_empty_mapping_is_accepted(self, service):
        assert service._prepare_props({}) == {}


# --- ingestion ------------------------------------------------------------


class TestIngestDocument:
    async def test_a_document_is_ingested(self, service):
        assert await service.ingest_document(a_document(), [a_chunk()]) is True

    async def test_the_document_node_carries_the_organization(self, service, session):
        """Neo4j has no TenantCollection; every statement has to carry it."""
        await service.ingest_document(a_document(), [])
        assert session.params[0]["org_id"] == ORG_ID

    async def test_every_chunk_carries_the_organization(self, service, session):
        await service.ingest_document(a_document(), [a_chunk(), a_chunk(id="c2")])
        assert all(p["org_id"] == ORG_ID for p in session.params)

    async def test_the_chunks_are_linked_to_their_document(self, service, session):
        await service.ingest_document(a_document(), [a_chunk()])
        assert "HAS_CHUNK" in session.queries

    async def test_the_metadata_is_serialised(self, service, session):
        await service.ingest_document(a_document(), [a_chunk()])
        assert isinstance(session.params[0]["metadata"], str)

    async def test_the_agent_scope_is_lifted_out_for_filtering(self, service, session):
        """It lives in the chunk metadata blob, which a query cannot filter
        on; the column is what makes agent-scoped retrieval possible."""
        await service.ingest_document(a_document(), [a_chunk()])
        assert session.params[1]["agent_id"] == "agent_1"

    async def test_the_filename_is_lifted_out_too(self, service, session):
        await service.ingest_document(a_document(), [a_chunk()])
        assert session.params[1]["filename"] == "handbook.pdf"

    async def test_a_chunk_without_metadata_is_accepted(self, service, session):
        await service.ingest_document(a_document(), [a_chunk(metadata={})])
        assert session.params[1]["agent_id"] is None

    async def test_a_missing_embedding_is_generated(self, service, session):
        """A chunk with no vector is invisible to search, so ingesting one
        would silently drop it from the knowledge graph."""
        await service.ingest_document(a_document(), [a_chunk(embedding=None)])
        assert session.params[1]["embedding"] == [0.1] * 8

    async def test_a_document_with_no_chunks_is_accepted(self, service, session):
        assert await service.ingest_document(a_document(), []) is True
        assert len(session.statements) == 1


class TestAddEntitiesAndRelationships:
    async def test_entities_are_added(self, service, session):
        entity = GraphEntity(id="e1", name="Kita", type="Product", properties={})
        assert await service.add_entities_and_relationships([entity], []) is True
        assert "Entity" in session.queries

    async def test_entities_carry_the_organization(self, service, session):
        entity = GraphEntity(id="e1", name="Kita", type="Product", properties={})
        await service.add_entities_and_relationships([entity], [])
        assert all(p["org_id"] == ORG_ID for p in session.params)

    async def test_relationships_are_added(self, service, session):
        rel = GraphRelationship(
            source_id="e1", target_id="e2", rel_type="RELATED_TO", properties={}
        )
        await service.add_entities_and_relationships([], [rel])
        assert session.statements

    async def test_nothing_to_add_is_accepted(self, service, session):
        assert await service.add_entities_and_relationships([], []) is True


# --- upserts --------------------------------------------------------------


class TestUpsertNode:
    async def test_an_id_is_required(self, service):
        """Without one the MERGE has nothing to match on and every call
        would create a new node."""
        with pytest.raises(ValueError):
            await service.upsert_node("Entity", {"name": "Kita"})

    async def test_a_node_is_upserted(self, monkeypatch, service):
        session = FakeSession(single_result=["e1"])
        service.driver.session = MagicMock(return_value=session)
        assert await service.upsert_node("Entity", {"id": "e1"}) == "e1"

    async def test_the_organization_is_part_of_the_key(self, service):
        session = FakeSession(single_result=["e1"])
        service.driver.session = MagicMock(return_value=session)
        await service.upsert_node("Entity", {"id": "e1"})
        assert session.params[0]["org_id"] == ORG_ID
        assert "org_id" in session.queries


class TestUpsertRelationship:
    async def test_a_relationship_is_upserted(self, service):
        session = FakeSession(single_result=[1])
        service.driver.session = MagicMock(return_value=session)
        assert await service.upsert_relationship("a", "b", "RELATED_TO", {}) is True

    async def test_no_match_reports_failure(self, service):
        session = FakeSession(single_result=[0])
        service.driver.session = MagicMock(return_value=session)
        assert await service.upsert_relationship("a", "b", "RELATED_TO", {}) is False

    async def test_both_ends_are_scoped_to_the_organization(self, service):
        """Otherwise a relationship could be drawn to another tenant's node."""
        session = FakeSession(single_result=[1])
        service.driver.session = MagicMock(return_value=session)
        await service.upsert_relationship("a", "b", "RELATED_TO", {})
        assert session.queries.count("org_id: $org_id") == 2


# --- lifecycle ------------------------------------------------------------


class TestLifecycle:
    async def test_the_driver_is_closed(self, service):
        await service.close()
        service.driver.close.assert_awaited_once()

    async def test_the_schema_is_initialised(self, service, session):
        await service.initialize_schema()
        assert "CREATE CONSTRAINT" in session.queries
        assert "CREATE INDEX" in session.queries

    async def test_the_constraints_are_scoped_per_organization(self, service, session):
        """A globally unique document id would let one tenant's ingest
        collide with another's."""
        await service.initialize_schema()
        assert "org_id" in session.queries

    async def test_a_missing_vector_index_capability_is_survivable(
        self, service, monkeypatch
    ):
        """The vector index needs Neo4j 5.11+; an older server should still
        get its constraints rather than failing setup outright."""

        class PartialSession(FakeSession):
            async def run(self, query, **params):
                if "VECTOR INDEX" in query:
                    raise RuntimeError("unsupported")
                return await super().run(query, **params)

        session = PartialSession()
        service.driver.session = MagicMock(return_value=session)
        await service.initialize_schema()
        assert "CREATE CONSTRAINT" in session.queries
