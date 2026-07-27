"""Tests for the agent-management, LLM and parse tools.

These are the tools the Agent Creator uses to build other agents. Unlike the
retrieval tools they reach straight into the database rather than taking a
service from deps, so the tenant scoping is theirs to get right -- an agent
created against the wrong organization would be invisible to the user who
asked for it and visible to one who did not.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.tools.agent_creation_tools import (
    create_agent,
    get_agent,
    list_agents,
    update_agent,
)
from app.services.tools.llm_tools import list_available_llms
from app.services.tools.parse_tools import fetch_latest_parse

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"


def a_context(**deps) -> SimpleNamespace:
    base = {"org_id": ORG_ID, "agent_id": "agent_1"}
    base.update(deps)
    return SimpleNamespace(deps=base, usage=MagicMock())


@pytest.fixture
def llms(patched_db, mongo_client):
    collection = mongo_client["kita_test_db"]["llms"]
    now = datetime.now(timezone.utc)
    collection.insert_one(
        {
            "org_id": ORG_ID,
            "name": "openrouter/x-ai/grok-4.3",
            "model": "x-ai/grok-4.3",
            "provider": "openrouter",
            "created_at": now,
            "updated_at": now,
        }
    )
    return collection


@pytest.fixture
def agents(patched_db, mongo_client):
    return mongo_client["kita_test_db"]["agents"]


# --- creating agents ------------------------------------------------------


class TestCreateAgent:
    async def test_an_agent_is_created(self, llms, agents):
        result = await create_agent(
            a_context(), "Scribe", "writer", "write things", "trained on prose"
        )
        assert "Successfully created" in result
        assert agents.count_documents({}) == 1

    async def test_it_is_scoped_to_the_calling_organization(self, llms, agents):
        """These tools resolve their own collections, so nothing upstream is
        enforcing the tenant for them."""
        await create_agent(a_context(), "Scribe", "writer", "goal", "backstory")
        assert agents.find_one({})["org_id"] == ORG_ID

    async def test_the_identity_is_stored(self, llms, agents):
        await create_agent(a_context(), "Scribe", "writer", "goal", "backstory")
        stored = agents.find_one({})
        assert stored["name"] == "Scribe" and stored["role"] == "writer"

    async def test_personalities_are_stored(self, llms, agents):
        await create_agent(
            a_context(), "Scribe", "r", "g", "b", personalities=["terse"]
        )
        assert agents.find_one({})["personalities"] == ["terse"]

    async def test_an_explicit_llm_is_used(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b", llm_id="llm_explicit")
        assert agents.find_one({})["llm_id"] == "llm_explicit"

    async def test_the_configured_default_model_is_chosen(
        self, llms, agents, monkeypatch
    ):
        """The creator agent rarely knows an LLM id, so leaving it out has to
        resolve to something usable rather than an empty string."""
        monkeypatch.setenv("LLM_MODEL", "x-ai/grok-4.3")
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        assert agents.find_one({})["llm_id"]

    async def test_the_null_string_is_treated_as_absent(self, llms, agents):
        """Models routinely pass the string "null" for an optional argument."""
        await create_agent(a_context(), "Scribe", "r", "g", "b", llm_id="null")
        assert agents.find_one({})["llm_id"] != "null"

    async def test_it_falls_back_to_grok_when_the_configured_model_is_absent(
        self, llms, agents, monkeypatch
    ):
        monkeypatch.setenv("LLM_MODEL", "not/registered")
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        assert agents.find_one({})["llm_id"]

    async def test_no_registered_llm_is_explained_not_raised(
        self, patched_db, agents, monkeypatch
    ):
        """The empty-string fallback tripped AgentCreateRequest's own
        validation, so an organization with no models registered got an
        opaque pydantic error instead of something the agent could relay."""
        monkeypatch.setenv("LLM_MODEL", "not/registered")
        result = await create_agent(a_context(), "Scribe", "r", "g", "b")
        assert "no LLM is registered" in result
        assert agents.count_documents({}) == 0

    async def test_the_available_models_are_offered(self, llms, agents, monkeypatch):
        """The creator agent's next move is to ask the user to pick one, so
        the message has to say what there is."""
        monkeypatch.setenv("LLM_MODEL", "not/registered")
        llms.delete_many({"model": "x-ai/grok-4.3"})
        llms.insert_one(
            {
                "org_id": ORG_ID,
                "name": "openrouter/other",
                "model": "vendor/other-model",
                "provider": "openrouter",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        result = await create_agent(a_context(), "Scribe", "r", "g", "b")
        assert "vendor/other-model" in result


class TestGetAgent:
    async def test_an_agent_is_described(self, llms, agents):
        await create_agent(a_context(), "Scribe", "writer", "goal", "backstory")
        base_id = agents.find_one({})["base_id"]
        result = await get_agent(a_context(), base_id)
        assert "Scribe" in result and "writer" in result

    async def test_a_missing_agent_is_reported_not_raised(self, llms, agents):
        assert "does not exist" in await get_agent(
            a_context(), "6a67000000000000000000ff"
        )

    async def test_a_pinned_version_is_resolved(self, llms, agents):
        await create_agent(a_context(), "Scribe", "writer", "goal", "backstory")
        base_id = agents.find_one({})["base_id"]
        assert "Scribe" in await get_agent(a_context(), f"{base_id}-v1")

    async def test_an_unknown_version_is_reported(self, llms, agents):
        await create_agent(a_context(), "Scribe", "writer", "goal", "backstory")
        base_id = agents.find_one({})["base_id"]
        assert "does not exist" in await get_agent(a_context(), f"{base_id}-v9")

    async def test_the_model_is_named(self, llms, agents):
        """The creator agent uses this to tell the user what an agent runs
        on, so an id alone is not enough."""
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        base_id = agents.find_one({})["base_id"]
        assert "x-ai/grok-4.3" in await get_agent(a_context(), base_id)

    async def test_a_malformed_model_id_does_not_stop_the_description(
        self, llms, agents
    ):
        """get_llm raises on anything that is not an ObjectId, and an agent
        can carry a stale id; describing the agent is the point of the tool."""
        await create_agent(a_context(), "Scribe", "r", "g", "b", llm_id="gone")
        base_id = agents.find_one({})["base_id"]
        result = await get_agent(a_context(), base_id)
        assert "Unknown LLM" in result and "Scribe" in result

    async def test_a_deleted_model_is_reported_as_unknown(self, llms, agents):
        from bson import ObjectId

        await create_agent(a_context(), "Scribe", "r", "g", "b", llm_id=str(ObjectId()))
        base_id = agents.find_one({})["base_id"]
        assert "Unknown LLM" in await get_agent(a_context(), base_id)

    async def test_another_organizations_agent_is_not_visible(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        base_id = agents.find_one({})["base_id"]
        intruder = SimpleNamespace(deps={"org_id": OTHER_ORG}, usage=MagicMock())
        assert "does not exist" in await get_agent(intruder, base_id)


class TestListAgents:
    async def test_no_agents_says_so(self, llms, agents):
        assert "No agents found" in await list_agents(a_context())

    async def test_the_agents_are_listed(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        await create_agent(a_context(), "Analyst", "r", "g", "b")
        result = await list_agents(a_context())
        assert "Scribe" in result and "Analyst" in result

    async def test_another_organizations_agents_are_not_listed(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        intruder = SimpleNamespace(deps={"org_id": OTHER_ORG}, usage=MagicMock())
        assert "No agents found" in await list_agents(intruder)


class TestUpdateAgent:
    async def test_an_agent_is_updated(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        base_id = agents.find_one({})["base_id"]
        assert "Successfully updated" in await update_agent(
            a_context(), base_id, name="Renamed"
        )

    async def test_a_missing_agent_is_reported_not_raised(self, llms, agents):
        result = await update_agent(a_context(), "6a67000000000000000000ff", name="x")
        assert "Failed to update" in result

    async def test_the_change_is_stored(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        base_id = agents.find_one({})["base_id"]
        await update_agent(a_context(), base_id, name="Renamed")
        assert agents.count_documents({"name": "Renamed"}) == 1

    async def test_a_new_version_is_written_by_default(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        base_id = agents.find_one({})["base_id"]
        await update_agent(a_context(), base_id, name="Renamed")
        assert agents.count_documents({}) == 2

    async def test_an_in_place_update_writes_no_new_version(self, llms, agents):
        """The creator agent iterates on a draft; a version per keystroke
        would bury the history."""
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        base_id = agents.find_one({})["base_id"]
        await update_agent(a_context(), base_id, name="Renamed", new_version=False)
        assert agents.count_documents({}) == 1

    async def test_another_organization_cannot_update_it(self, llms, agents):
        await create_agent(a_context(), "Scribe", "r", "g", "b")
        base_id = agents.find_one({})["base_id"]
        intruder = SimpleNamespace(deps={"org_id": OTHER_ORG}, usage=MagicMock())
        assert "Failed to update" in await update_agent(intruder, base_id, name="x")
        assert agents.count_documents({"name": "Scribe"}) == 1


# --- LLMs -----------------------------------------------------------------


class TestListAvailableLlms:
    async def test_the_models_are_listed(self, llms):
        result = await list_available_llms(a_context())
        assert "x-ai/grok-4.3" in result

    async def test_no_models_says_so(self, patched_db):
        assert "No LLMs found" in await list_available_llms(a_context())

    async def test_another_organizations_models_are_not_listed(self, llms):
        intruder = SimpleNamespace(deps={"org_id": OTHER_ORG}, usage=MagicMock())
        assert "No LLMs found" in await list_available_llms(intruder)

    async def test_the_id_is_included(self, llms):
        """The creator agent has to pass one back into create_agent."""
        assert "ID:" in await list_available_llms(a_context())


# --- parse results --------------------------------------------------------


class TestFetchLatestParse:
    async def test_the_parse_result_is_returned(self):
        service = MagicMock()
        service.get_latest_parse = AsyncMock(
            return_value=SimpleNamespace(result={"pages": []})
        )
        assert await fetch_latest_parse(a_context(parse_service=service), "file_1") == {
            "pages": []
        }

    async def test_a_missing_service_is_reported_not_raised(self):
        assert "error" in await fetch_latest_parse(a_context(), "file_1")

    async def test_an_unparsed_file_is_reported(self):
        service = MagicMock()
        service.get_latest_parse = AsyncMock(return_value=None)
        result = await fetch_latest_parse(a_context(parse_service=service), "file_1")
        assert "No parse records" in result["error"]

    async def test_a_failure_is_reported_not_raised(self):
        service = MagicMock()
        service.get_latest_parse = AsyncMock(side_effect=RuntimeError("mongo down"))
        result = await fetch_latest_parse(a_context(parse_service=service), "file_1")
        assert "Error fetching" in result["error"]
