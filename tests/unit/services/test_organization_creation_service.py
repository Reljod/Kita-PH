"""Tests for app.services.organization_creation_service.

Scaffolding a new organization is a multi-step write across four collections
with no transaction around it. What matters is that a half-finished run does
not leave the org looking ready: the status has to end up "failed" and the
partial records have to go.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.db import TenantCollection
from app.services.organization_creation_service import OrganizationCreationService

ORG_ID = "org_test_0001"


@pytest.fixture
def agents_collection(mongo_db):
    return mongo_db["agents"]


@pytest.fixture
def llm_service(object_id) -> MagicMock:
    service = MagicMock(name="llm_service")
    service.add_llm.return_value = MagicMock(id="llm_1")
    service.collection = MagicMock()
    return service


@pytest.fixture
def agent_service(mongo_db, agents_collection) -> MagicMock:
    service = MagicMock(name="agent_service")
    service.collection = TenantCollection(agents_collection, ORG_ID)
    return service


@pytest.fixture
def tool_service() -> MagicMock:
    service = MagicMock(name="tool_service")
    service.register_tool = AsyncMock()
    service.get_tool_by_name = AsyncMock(
        side_effect=lambda name: {"_id": ObjectId(), "name": name}
    )
    service.collection = MagicMock()
    return service


@pytest.fixture
def rag_service() -> MagicMock:
    service = MagicMock(name="rag_service")
    service.add_rag = AsyncMock(return_value=MagicMock(id="rag_1"))
    service.update_embedding = AsyncMock()
    service.collection = MagicMock()
    return service


@pytest.fixture
def org_service() -> MagicMock:
    service = MagicMock(name="org_service")
    service.get_org.return_value = MagicMock(org_name="Acme", org_code="ACME01")
    service.update_org_status = MagicMock()
    return service


@pytest.fixture
def service(
    llm_service, agent_service, tool_service, rag_service, org_service
) -> OrganizationCreationService:
    return OrganizationCreationService(
        llm_service, agent_service, tool_service, rag_service, org_service
    )


# --- the happy path -------------------------------------------------------


class TestInitializeOrg:
    async def test_the_org_is_marked_completed(self, service, org_service):
        await service.initialize_org(ORG_ID)
        org_service.update_org_status.assert_called_once_with(ORG_ID, "completed")

    async def test_a_default_llm_is_created(self, service, llm_service):
        await service.initialize_org(ORG_ID)
        llm_service.add_llm.assert_called_once()

    async def test_the_tools_are_registered(self, service, tool_service):
        await service.initialize_org(ORG_ID)
        assert tool_service.register_tool.await_count > 0

    async def test_both_default_agents_are_seeded(self, service, agents_collection):
        await service.initialize_org(ORG_ID)
        names = sorted(d["name"] for d in agents_collection.find({}))
        assert names == ["Agent Creator", "Kita Assistant"]

    async def test_the_organization_details_become_a_memory(self, service, rag_service):
        await service.initialize_org(ORG_ID)
        rag_service.add_rag.assert_awaited_once()

    async def test_nothing_is_reverted_on_success(self, service, llm_service):
        await service.initialize_org(ORG_ID)
        llm_service.collection.delete_many.assert_not_called()


# --- failure and revert ---------------------------------------------------


class TestInitializeOrgFailure:
    async def test_the_org_is_marked_failed(self, service, org_service, tool_service):
        tool_service.register_tool = AsyncMock(side_effect=RuntimeError("mongo down"))
        with pytest.raises(RuntimeError):
            await service.initialize_org(ORG_ID)
        org_service.update_org_status.assert_called_once_with(ORG_ID, "failed")

    async def test_the_original_error_is_re_raised(self, service, tool_service):
        """The caller decides whether to retry, so it needs the real cause and
        not a generic scaffolding error."""
        tool_service.register_tool = AsyncMock(side_effect=RuntimeError("mongo down"))
        with pytest.raises(RuntimeError, match="mongo down"):
            await service.initialize_org(ORG_ID)

    async def test_the_partial_scaffolding_is_removed(
        self, service, tool_service, llm_service, rag_service
    ):
        """A half-seeded org that still has an LLM and some tools looks ready
        to the UI while being unusable."""
        rag_service.add_rag = AsyncMock(side_effect=RuntimeError("embedding failed"))
        with pytest.raises(RuntimeError):
            await service.initialize_org(ORG_ID)
        llm_service.collection.delete_many.assert_called_once_with({})
        tool_service.collection.delete_many.assert_called_once_with({})
        rag_service.collection.delete_many.assert_called_once_with({})

    async def test_the_seeded_agents_are_removed(
        self, service, rag_service, agents_collection
    ):
        rag_service.add_rag = AsyncMock(side_effect=RuntimeError("embedding failed"))
        with pytest.raises(RuntimeError):
            await service.initialize_org(ORG_ID)
        assert agents_collection.count_documents({}) == 0

    async def test_a_failing_revert_does_not_mask_the_original_error(
        self, service, rag_service, llm_service
    ):
        """If cleanup also fails there is nothing useful to say about it, and
        swallowing the first error would hide why the run failed at all."""
        rag_service.add_rag = AsyncMock(side_effect=RuntimeError("embedding failed"))
        llm_service.collection.delete_many.side_effect = RuntimeError("revert failed")
        with pytest.raises(RuntimeError, match="embedding failed"):
            await service.initialize_org(ORG_ID)

    async def test_the_status_is_still_marked_failed_when_revert_fails(
        self, service, rag_service, llm_service, org_service
    ):
        rag_service.add_rag = AsyncMock(side_effect=RuntimeError("embedding failed"))
        llm_service.collection.delete_many.side_effect = RuntimeError("revert failed")
        with pytest.raises(RuntimeError):
            await service.initialize_org(ORG_ID)
        org_service.update_org_status.assert_called_once_with(ORG_ID, "failed")


# --- the default LLM ------------------------------------------------------


class TestGenerateDefaultLlm:
    def test_the_configured_model_is_used(self, service, llm_service, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "x-ai/grok-4.3")
        service.generate_default_llm()
        assert llm_service.add_llm.call_args[0][0].model == "x-ai/grok-4.3"

    def test_the_name_is_namespaced_by_provider(
        self, service, llm_service, monkeypatch
    ):
        monkeypatch.setenv("LLM_MODEL", "x-ai/grok-4.3")
        service.generate_default_llm()
        assert llm_service.add_llm.call_args[0][0].name == "openrouter/x-ai/grok-4.3"

    def test_an_already_namespaced_model_is_not_doubled(
        self, service, llm_service, monkeypatch
    ):
        """ "openrouter/openrouter/..." would not resolve at the provider."""
        monkeypatch.setenv("LLM_MODEL", "openrouter/x-ai/grok-4.3")
        service.generate_default_llm()
        request = llm_service.add_llm.call_args[0][0]
        assert request.name == "openrouter/x-ai/grok-4.3"
        assert request.model == "x-ai/grok-4.3"

    def test_the_provider_is_openrouter(self, service, llm_service):
        service.generate_default_llm()
        assert llm_service.add_llm.call_args[0][0].provider == "openrouter"

    def test_the_new_llm_id_is_returned(self, service):
        assert service.generate_default_llm() == "llm_1"


# --- the default agents ---------------------------------------------------


class TestGenerateDefaultAgents:
    async def test_the_agents_are_scoped_to_the_organization(
        self, service, agents_collection
    ):
        await service.generate_default_agents("llm_1")
        assert all(d["org_id"] == ORG_ID for d in agents_collection.find({}))

    async def test_each_agent_is_its_own_base(self, service, agents_collection):
        """base_id has to be set at insert time here; unlike create_agent
        there is no second write to backfill it."""
        await service.generate_default_agents("llm_1")
        for doc in agents_collection.find({}):
            assert doc["base_id"] == str(doc["_id"])

    async def test_the_agents_start_at_version_one(self, service, agents_collection):
        await service.generate_default_agents("llm_1")
        assert all(d["version"] == 1 for d in agents_collection.find({}))

    async def test_editing_a_seeded_agent_does_not_reuse_its_version(
        self, service, agents_collection, mongo_db
    ):
        """These docs are inserted straight into Mongo without touching the
        version counter, so the counter has to fast-forward past them exactly
        as it does for create_agent."""
        from app.models.agent import AgentUpdateRequest
        from app.services.agent_service import AgentService

        await service.generate_default_agents("llm_1")
        real = AgentService(MagicMock(), TenantCollection(agents_collection, ORG_ID))
        base_id = agents_collection.find_one({"name": "Kita Assistant"})["base_id"]
        updated = await real.update_agent(base_id, AgentUpdateRequest(name="Renamed"))
        assert updated.version == 2

    async def test_both_agents_share_the_llm(self, service, agents_collection):
        await service.generate_default_agents("llm_x")
        assert all(d["llm_id"] == "llm_x" for d in agents_collection.find({}))

    async def test_the_assistant_gets_its_retrieval_tools(
        self, service, agents_collection
    ):
        await service.generate_default_agents("llm_1")
        kita = agents_collection.find_one({"name": "Kita Assistant"})
        assert len(kita["tools"]) == 3

    async def test_the_creator_gets_the_agent_management_tools(
        self, service, agents_collection
    ):
        await service.generate_default_agents("llm_1")
        creator = agents_collection.find_one({"name": "Agent Creator"})
        assert len(creator["tools"]) == 7

    async def test_unregistered_tools_are_skipped_rather_than_stored_as_none(
        self, service, tool_service, agents_collection
    ):
        """A None in the tools list breaks every later tool-name resolution."""
        tool_service.get_tool_by_name = AsyncMock(return_value=None)
        await service.generate_default_agents("llm_1")
        assert all(d["tools"] == [] for d in agents_collection.find({}))

    async def test_the_backstories_are_loaded(self, service, agents_collection):
        await service.generate_default_agents("llm_1")
        assert all(d["backstory"].strip() for d in agents_collection.find({}))


# --- tools and memories ---------------------------------------------------


class TestGenerateDefaultTools:
    async def test_every_available_tool_is_registered(self, service, tool_service):
        from app.services.tools import get_available_tools

        await service.generate_default_tools()
        assert tool_service.register_tool.await_count == len(get_available_tools())


class TestInitializeDefaultGlobalMemories:
    async def test_the_memory_names_the_organization(self, service, rag_service):
        await service.initialize_default_global_memories(ORG_ID)
        content = rag_service.add_rag.await_args[0][0].content
        assert "Acme" in content and "ACME01" in content

    async def test_the_memory_is_organization_wide(self, service, rag_service):
        """agent_id=None is what makes it visible to every agent in the org."""
        await service.initialize_default_global_memories(ORG_ID)
        assert rag_service.add_rag.await_args[0][0].agent_id is None

    async def test_the_embedding_is_generated_before_the_org_goes_live(
        self, service, rag_service
    ):
        """Deferring it would leave the first search returning nothing."""
        await service.initialize_default_global_memories(ORG_ID)
        rag_service.update_embedding.assert_awaited_once_with("rag_1")

    async def test_a_missing_organization_writes_no_memory(
        self, service, org_service, rag_service
    ):
        org_service.get_org.return_value = None
        await service.initialize_default_global_memories(ORG_ID)
        rag_service.add_rag.assert_not_awaited()


class TestRevertInitialization:
    def test_every_scaffolded_collection_is_cleared(
        self, service, llm_service, tool_service, rag_service, agents_collection
    ):
        agents_collection.insert_one({"org_id": ORG_ID, "name": "x"})
        service.revert_initialization()
        assert agents_collection.count_documents({}) == 0
        llm_service.collection.delete_many.assert_called_once_with({})
        tool_service.collection.delete_many.assert_called_once_with({})
        rag_service.collection.delete_many.assert_called_once_with({})

    def test_another_organizations_records_survive(self, service, agents_collection):
        """The deletes go through TenantCollection, so an empty filter means
        "everything in this org" rather than everything."""
        agents_collection.insert_one({"org_id": "org_other", "name": "theirs"})
        service.revert_initialization()
        assert agents_collection.count_documents({}) == 1
