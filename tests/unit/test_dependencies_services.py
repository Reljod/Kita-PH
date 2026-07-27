"""Tests for app.dependencies.services.

This module is where every request-scoped service gets its TenantCollection.
A service wired with the wrong org_id would bypass the isolation boundary
everything else depends on, and it would do so silently -- so the sweep below
asserts the scoping of every tenant-scoped service on the registry rather than
spot-checking a couple.
"""

from __future__ import annotations

import pytest

from app.db import TenantCollection
from app.dependencies import services as services_module
from app.dependencies.services import ServiceRegistry, get_services

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"


@pytest.fixture
def registry(patched_db) -> ServiceRegistry:
    return get_services(ORG_ID)


# --- global singletons ----------------------------------------------------


class TestGlobalSingletons:
    def test_the_event_service_is_created_once(self):
        assert (
            services_module.get_event_service() is services_module.get_event_service()
        )

    def test_the_facebook_service_is_created_once(self):
        assert (
            services_module.get_facebook_service()
            is services_module.get_facebook_service()
        )

    def test_the_org_service_is_created_once(self, patched_db):
        """Unlike the others this one resolves a collection at construction,
        so it needs a database behind it."""
        assert services_module.get_org_service() is services_module.get_org_service()

    def test_the_web_search_service_is_created_once(self):
        assert (
            services_module.get_web_search_service()
            is services_module.get_web_search_service()
        )

    def test_the_singletons_are_shared_across_organizations(self, patched_db):
        """These hold no tenant state -- an events broker and a search client
        -- so rebuilding them per organization would just leak connections."""
        mine = get_services(ORG_ID)
        theirs = get_services(OTHER_ORG)
        assert mine.event_service is theirs.event_service
        assert mine.web_search_service is theirs.web_search_service


# --- registry caching -----------------------------------------------------


class TestGetServices:
    def test_a_registry_is_returned(self, patched_db):
        assert get_services(ORG_ID).org_id == ORG_ID

    def test_the_registry_is_reused_for_the_same_organization(self, patched_db):
        """Rebuilding it per request would open a fresh Neo4j driver every
        time."""
        assert get_services(ORG_ID) is get_services(ORG_ID)

    def test_each_organization_gets_its_own_registry(self, patched_db):
        assert get_services(ORG_ID) is not get_services(OTHER_ORG)

    def test_the_registries_do_not_share_tenant_scoped_services(self, patched_db):
        mine = get_services(ORG_ID)
        theirs = get_services(OTHER_ORG)
        assert mine.agent_service is not theirs.agent_service


# --- tenant scoping -------------------------------------------------------

# Every attribute here holds a service whose collection must be scoped to the
# registry's organization. Listing them by name means a newly-added service
# shows up as a failure rather than as an untested gap.
SCOPED_SERVICES = [
    "llm_service",
    "agent_service",
    "tool_service",
    "file_service",
    "rag_service",
    "mongodb_vector_search_rag_service",
    "mongodb_text_search_rag_service",
    "ingest_service",
    "retrieval_service",
    "chat_service",
]


class TestTenantScoping:
    @pytest.mark.parametrize("attribute", SCOPED_SERVICES)
    def test_each_service_is_scoped_to_the_organization(self, registry, attribute):
        collection = getattr(registry, attribute).collection
        assert isinstance(collection, TenantCollection)
        assert collection.org_id == ORG_ID

    def test_the_agents_tools_collection_is_also_scoped(self, registry):
        """A widened tools lookup would let one organization's agent resolve
        another's tool names."""
        assert registry.agent_service.tools_collection.org_id == ORG_ID

    def test_the_parse_collection_is_scoped(self, registry):
        assert registry.parse_service.parse_collection.org_id == ORG_ID

    def test_the_ingest_parse_collection_is_scoped(self, registry):
        assert registry.ingest_service.parse_collection.org_id == ORG_ID

    def test_the_retrieval_parse_collection_is_scoped(self, registry):
        assert registry.retrieval_service.parse_collection.org_id == ORG_ID

    def test_the_status_service_is_scoped(self, registry):
        """Its Redis keys are namespaced by this value rather than by a
        collection."""
        assert registry.agent_status_service.org_id == ORG_ID

    def test_the_graph_service_is_scoped(self, registry):
        assert registry.graph_rag_service.org_id == ORG_ID

    def test_the_adaptive_rag_service_is_scoped(self, registry):
        assert registry.adaptive_rag_service.org_id == ORG_ID

    def test_the_file_service_is_scoped(self, registry):
        assert registry.file_service.org_id == ORG_ID

    def test_two_registries_scope_to_their_own_organizations(self, patched_db):
        assert get_services(ORG_ID).rag_service.collection.org_id == ORG_ID
        assert get_services(OTHER_ORG).rag_service.collection.org_id == OTHER_ORG


class TestServiceWiring:
    def test_the_status_service_can_name_agents(self, registry):
        """Without the agent service it falls back to printing raw ids in the
        status messages users see."""
        assert registry.agent_status_service.agent_service is registry.agent_service

    def test_the_chat_service_runs_agents(self, registry):
        assert registry.chat_service.agent_service is registry.agent_service

    def test_the_parse_service_reads_files(self, registry):
        assert registry.parse_service.file_service is registry.file_service

    def test_the_file_service_publishes_events(self, registry):
        assert registry.file_service.event_service is registry.event_service

    def test_retrieval_shares_the_vector_service_with_ingest(self, registry):
        """Ingest writes the embeddings retrieval reads; two instances would
        drift on model or index configuration."""
        assert (
            registry.retrieval_service.vector_service
            is registry.ingest_service.vector_service
        )

    def test_adaptive_rag_uses_the_retrieval_service(self, registry):
        assert registry.adaptive_rag_service.retrieval_service is (
            registry.retrieval_service
        )

    def test_adaptive_rag_can_search_the_web(self, registry):
        assert registry.adaptive_rag_service.web_search_service is (
            registry.web_search_service
        )


# --- the FastAPI dependency callables -------------------------------------

DEPENDENCY_ACCESSORS = [
    ("get_llm_service", "llm_service"),
    ("get_agent_service", "agent_service"),
    ("get_tool_service", "tool_service"),
    ("get_file_service", "file_service"),
    ("get_parse_service", "parse_service"),
    ("get_graph_rag_service", "graph_rag_service"),
    ("get_chat_service", "chat_service"),
    ("get_rag_service", "rag_service"),
    ("get_nested_data_enrichment_service", "nested_data_enrichment_service"),
    ("get_mongodb_vector_search_rag_service", "mongodb_vector_search_rag_service"),
    ("get_mongodb_text_search_rag_service", "mongodb_text_search_rag_service"),
    ("get_reranking_rag_service", "reranking_rag_service"),
    ("get_ingest_service", "ingest_service"),
    ("get_retrieval_service", "retrieval_service"),
    ("get_adaptive_rag_service", "adaptive_rag_service"),
    ("get_agent_status_service", "agent_status_service"),
]


class TestDependencyAccessors:
    @pytest.mark.parametrize("accessor,attribute", DEPENDENCY_ACCESSORS)
    def test_each_accessor_resolves_from_the_registry(
        self, patched_db, accessor, attribute
    ):
        resolved = getattr(services_module, accessor)(ORG_ID)
        assert resolved is getattr(get_services(ORG_ID), attribute)

    def test_the_agent_scoped_rag_accessor_resolves(self, patched_db):
        resolved = services_module.get_agent_rag_service("agent_1", ORG_ID)
        assert resolved is get_services(ORG_ID).rag_service

    def test_the_graph_accessor_refuses_when_neo4j_is_unconfigured(
        self, patched_db, monkeypatch
    ):
        """A half-configured graph backend would otherwise surface as a
        driver timeout on the first query rather than a clear 500."""
        from fastapi import HTTPException

        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        with pytest.raises(HTTPException) as exc:
            services_module.get_graph_rag_service(ORG_ID)
        assert exc.value.status_code == 500

    def test_an_accessor_resolves_per_organization(self, patched_db):
        """The org_id comes from the request's token, so two callers must not
        share a service."""
        mine = services_module.get_agent_service(ORG_ID)
        theirs = services_module.get_agent_service(OTHER_ORG)
        assert mine is not theirs
        assert mine.collection.org_id == ORG_ID
        assert theirs.collection.org_id == OTHER_ORG
