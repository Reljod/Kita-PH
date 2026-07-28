"""Tests for app.services.agent_service.

Agents are immutable-by-default: an edit writes a new document rather than
overwriting the old one, and clients can pin `<base_id>-v<n>` to keep talking
to a definition they have already validated. That versioning is the part worth
testing hard — a collision there silently changes an agent's behaviour under a
caller who explicitly asked for stability.

A real in-memory Mongo is used rather than a mock, because the sort/group/
upsert semantics are the behaviour under test.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from app.db import TenantCollection
from app.models.agent import AgentCreateRequest, AgentLanguage, AgentUpdateRequest
from app.services.agent_service import AgentService

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"


def a_request(**overrides) -> AgentCreateRequest:
    payload = {
        "name": "Researcher",
        "role": "analyst",
        "goal": "find things",
        "backstory": "trained on the archive",
        "llm_id": "llm_1",
    }
    payload.update(overrides)
    return AgentCreateRequest(**payload)


@pytest.fixture
def agents_collection(mongo_db):
    return mongo_db["agents"]


@pytest.fixture
def tools_collection(mongo_db):
    return mongo_db["tools"]


@pytest.fixture
def llm_service() -> MagicMock:
    service = MagicMock(name="llm_service")
    service.get_llm.return_value = MagicMock(model="openai/gpt-4o-mini")
    return service


@pytest.fixture
def service(llm_service, agents_collection) -> AgentService:
    return AgentService(llm_service, TenantCollection(agents_collection, ORG_ID))


@pytest.fixture
def service_with_tools(
    llm_service, agents_collection, tools_collection
) -> AgentService:
    return AgentService(
        llm_service,
        TenantCollection(agents_collection, ORG_ID),
        TenantCollection(tools_collection, ORG_ID),
    )


@pytest.fixture
async def agent(service):
    return await service.create_agent(a_request())


# --- creation -------------------------------------------------------------


class TestCreateAgent:
    async def test_an_agent_is_created_at_version_one(self, service):
        assert (await service.create_agent(a_request())).version == 1

    async def test_the_identity_fields_round_trip(self, service):
        created = await service.create_agent(a_request(name="Scribe", role="writer"))
        assert created.name == "Scribe" and created.role == "writer"

    async def test_the_base_id_is_backfilled_from_the_inserted_id(self, service, agent):
        """base_id is what every later lookup keys on, but it is only knowable
        after the insert — so creation is a two-step write."""
        assert agent.base_id == agent.id
        assert ObjectId(agent.base_id)

    async def test_the_base_id_is_persisted_not_just_returned(
        self, service, agent, agents_collection
    ):
        stored = agents_collection.find_one({"_id": ObjectId(agent.base_id)})
        assert stored["base_id"] == agent.base_id

    async def test_the_agent_is_scoped_to_the_organization(
        self, service, agent, agents_collection
    ):
        assert (
            agents_collection.find_one({"base_id": agent.base_id})["org_id"] == ORG_ID
        )

    async def test_a_system_prompt_is_built_at_creation(self, service, agent):
        assert agent.system_prompt and "Researcher" in agent.system_prompt

    async def test_personalities_are_stored(self, service):
        created = await service.create_agent(a_request(personalities=["curious"]))
        assert created.personalities == ["curious"]

    async def test_tools_default_to_empty(self, service, agent):
        assert agent.tools == []

    async def test_requested_tools_are_stored(self, service):
        created = await service.create_agent(a_request(tools=["web_search"]))
        assert created.tools == ["web_search"]

    async def test_two_agents_get_distinct_base_ids(self, service):
        first = await service.create_agent(a_request())
        second = await service.create_agent(a_request())
        assert first.base_id != second.base_id


# --- versioning -----------------------------------------------------------


class TestVersioning:
    async def test_an_update_allocates_the_next_version(self, service, agent):
        assert (
            await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        ).version == 2

    async def test_versions_never_collide_with_the_created_version(
        self, service, agent, agents_collection
    ):
        """The counter starts un-seeded while creation already used version 1.
        Handing 1 back out would put two documents at (base_id, v1) and make a
        pinned id resolve to either of them."""
        await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        versions = sorted(d["version"] for d in agents_collection.find({}))
        assert versions == [1, 2]

    async def test_versions_keep_climbing_across_repeated_edits(self, service, agent):
        for _ in range(4):
            await service.update_agent(agent.id, AgentUpdateRequest(name="x"))
        latest = service.get_agent(agent.id)
        assert latest.version == 5

    async def test_a_pinned_version_keeps_returning_its_own_definition(
        self, service, agent
    ):
        """This is the whole point of pinning: an edit must not reach back and
        change what an already-validated version says."""
        await service.update_agent(agent.id, AgentUpdateRequest(name="Changed"))
        assert service.get_agent(f"{agent.base_id}-v1").name == "Researcher"

    async def test_the_bare_id_resolves_to_the_latest_version(self, service, agent):
        await service.update_agent(agent.id, AgentUpdateRequest(name="Changed"))
        assert service.get_agent(agent.base_id).name == "Changed"

    async def test_the_counter_is_shared_across_every_versioning_path(
        self, service, agent, agents_collection
    ):
        """update_agent, add_tools and remove_tools all allocate from the same
        counter, so interleaving them must not repeat a number."""
        await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        await service.add_tools(agent.id, ["t1"])
        await service.remove_tools(agent.id, ["t1"])
        versions = sorted(d["version"] for d in agents_collection.find({}))
        assert versions == [1, 2, 3, 4]

    async def test_each_base_id_versions_independently(self, service):
        first = await service.create_agent(a_request())
        second = await service.create_agent(a_request())
        await service.update_agent(first.id, AgentUpdateRequest(name="B"))
        assert service.get_agent(second.id).version == 1


class TestParseAgentId:
    async def test_an_unknown_id_yields_nothing(self, service):
        assert service.get_agent(str(ObjectId())) is None

    async def test_an_unknown_version_of_a_known_agent_yields_nothing(
        self, service, agent
    ):
        assert service.get_agent(f"{agent.base_id}-v99") is None

    async def test_a_non_numeric_suffix_is_treated_as_part_of_the_id(self, service):
        """`parse_agent_id` only splits on a numeric suffix, so an id that
        merely contains "-v" is not mistaken for a pinned version."""
        assert service.get_agent("some-vendor-agent") is None


# --- reads ----------------------------------------------------------------


class TestGetAgent:
    async def test_an_agent_is_returned(self, service, agent):
        assert service.get_agent(agent.id).id == agent.id

    async def test_the_prompt_is_rebuilt_on_read(self, service, agent):
        """The prompt is not persisted, so a template change reaches existing
        agents without a migration."""
        assert "Researcher" in service.get_agent(agent.id).system_prompt

    async def test_another_organization_cannot_read_the_agent(
        self, llm_service, agents_collection, agent
    ):
        """TenantCollection is the whole isolation boundary — nothing below it
        knows about orgs."""
        intruder = AgentService(
            llm_service, TenantCollection(agents_collection, OTHER_ORG)
        )
        assert intruder.get_agent(agent.id) is None


class TestGetAllAgents:
    async def test_no_agents_yields_an_empty_list(self, service):
        assert service.get_all_agents() == []

    async def test_every_agent_is_listed(self, service):
        await service.create_agent(a_request(name="A"))
        await service.create_agent(a_request(name="B"))
        assert len(service.get_all_agents()) == 2

    async def test_only_the_latest_version_of_each_agent_is_listed(
        self, service, agent
    ):
        """Otherwise every edit would add a duplicate row to the UI."""
        await service.update_agent(agent.id, AgentUpdateRequest(name="Changed"))
        listed = service.get_all_agents()
        assert len(listed) == 1 and listed[0].name == "Changed"

    async def test_the_prompt_is_omitted_from_the_list_view(self, service, agent):
        """Building it per row would mean a template render and a tools lookup
        for every agent on the dashboard."""
        assert service.get_all_agents()[0].system_prompt is None

    async def test_another_organizations_agents_are_not_listed(
        self, llm_service, agents_collection, agent
    ):
        intruder = AgentService(
            llm_service, TenantCollection(agents_collection, OTHER_ORG)
        )
        assert intruder.get_all_agents() == []


# --- updates --------------------------------------------------------------


class TestUpdateAgent:
    async def test_updating_a_missing_agent_yields_nothing(self, service):
        assert (
            await service.update_agent(str(ObjectId()), AgentUpdateRequest(name="B"))
            is None
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("name", "Renamed"),
            ("role", "editor"),
            ("goal", "new goal"),
            ("backstory", "new backstory"),
            ("llm_id", "llm_2"),
            ("personalities", ["terse"]),
            ("tools", ["web_search"]),
            ("language", AgentLanguage.FILIPINO),
        ],
    )
    async def test_each_field_can_be_updated(self, service, agent, field, value):
        updated = await service.update_agent(
            agent.id, AgentUpdateRequest(**{field: value})
        )
        assert getattr(updated, field) == value

    async def test_unspecified_fields_are_carried_forward(self, service, agent):
        """A partial update must not blank out the fields it does not mention."""
        updated = await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        assert updated.role == "analyst" and updated.goal == "find things"

    async def test_the_creation_timestamp_is_carried_forward(self, service, agent):
        """A new version is the same agent, so its creation date should not
        jump forward every time someone edits it. Both sides are read back
        through Mongo because it stores millisecond precision."""
        updated = await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        original = service.get_agent(f"{agent.base_id}-v1")
        assert updated.created_at == original.created_at

    async def test_the_update_timestamp_moves(self, service, agent):
        updated = await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        assert updated.updated_at >= agent.updated_at

    async def test_the_prompt_is_rebuilt_from_the_new_values(self, service, agent):
        updated = await service.update_agent(
            agent.id, AgentUpdateRequest(name="Renamed")
        )
        assert "Renamed" in updated.system_prompt

    async def test_an_empty_update_still_produces_a_version(self, service, agent):
        assert (await service.update_agent(agent.id, AgentUpdateRequest())).version == 2


class TestUpdateAgentInPlace:
    """`new_version=False` edits the current document instead of branching."""

    async def test_no_new_document_is_written(self, service, agent, agents_collection):
        await service.update_agent(
            agent.id, AgentUpdateRequest(name="B"), new_version=False
        )
        assert agents_collection.count_documents({}) == 1

    async def test_the_version_does_not_move(self, service, agent):
        updated = await service.update_agent(
            agent.id, AgentUpdateRequest(name="B"), new_version=False
        )
        assert updated.version == 1

    async def test_the_change_is_visible(self, service, agent):
        await service.update_agent(
            agent.id, AgentUpdateRequest(name="B"), new_version=False
        )
        assert service.get_agent(agent.id).name == "B"

    async def test_unmentioned_fields_are_left_alone(self, service, agent):
        await service.update_agent(
            agent.id, AgentUpdateRequest(name="B"), new_version=False
        )
        assert service.get_agent(agent.id).role == "analyst"

    async def test_updating_a_missing_agent_yields_nothing(self, service):
        assert (
            await service.update_agent(
                str(ObjectId()), AgentUpdateRequest(name="B"), new_version=False
            )
            is None
        )

    async def test_it_edits_the_latest_version_not_the_pinned_one(self, service, agent):
        """Addressing an old version in place would rewrite history for anyone
        pinned to it, so the write lands on the head."""
        await service.update_agent(agent.id, AgentUpdateRequest(name="v2"))
        await service.update_agent(
            f"{agent.base_id}-v1", AgentUpdateRequest(goal="edited"), new_version=False
        )
        assert service.get_agent(f"{agent.base_id}-v1").goal == "find things"
        assert service.get_agent(agent.base_id).goal == "edited"


# --- deletion -------------------------------------------------------------


class TestDeleteAgent:
    async def test_an_agent_is_deleted(self, service, agent):
        assert service.delete_agent(agent.id) is True
        assert service.get_agent(agent.id) is None

    async def test_every_version_is_removed(self, service, agent, agents_collection):
        """Leaving old versions behind would let a pinned id keep resolving
        after the agent was deleted."""
        await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        service.delete_agent(agent.id)
        assert agents_collection.count_documents({}) == 0

    async def test_deleting_by_a_pinned_id_still_removes_the_whole_agent(
        self, service, agent, agents_collection
    ):
        await service.update_agent(agent.id, AgentUpdateRequest(name="B"))
        service.delete_agent(f"{agent.base_id}-v1")
        assert agents_collection.count_documents({}) == 0

    async def test_deleting_a_missing_agent_reports_failure(self, service):
        assert service.delete_agent(str(ObjectId())) is False

    async def test_another_organization_cannot_delete_the_agent(
        self, llm_service, agents_collection, agent
    ):
        intruder = AgentService(
            llm_service, TenantCollection(agents_collection, OTHER_ORG)
        )
        assert intruder.delete_agent(agent.id) is False
        assert agents_collection.count_documents({}) == 1


# --- tool attachment ------------------------------------------------------


class TestAddTools:
    async def test_a_tool_is_attached(self, service, agent):
        assert await service.add_tools(agent.id, ["t1"]) is True
        assert service.get_agent(agent.id).tools == ["t1"]

    async def test_attaching_produces_a_new_version(self, service, agent):
        await service.add_tools(agent.id, ["t1"])
        assert service.get_agent(agent.id).version == 2

    async def test_existing_tools_are_kept(self, service):
        created = await service.create_agent(a_request(tools=["t1"]))
        await service.add_tools(created.id, ["t2"])
        assert service.get_agent(created.id).tools == ["t1", "t2"]

    async def test_a_duplicate_tool_is_not_added_twice(self, service):
        created = await service.create_agent(a_request(tools=["t1"]))
        await service.add_tools(created.id, ["t1"])
        assert service.get_agent(created.id).tools == ["t1"]

    async def test_several_tools_can_be_attached_at_once(self, service, agent):
        await service.add_tools(agent.id, ["t1", "t2"])
        assert service.get_agent(agent.id).tools == ["t1", "t2"]

    async def test_attaching_to_a_missing_agent_reports_failure(self, service):
        assert await service.add_tools(str(ObjectId()), ["t1"]) is False

    async def test_the_identity_is_carried_into_the_new_version(self, service, agent):
        await service.add_tools(agent.id, ["t1"])
        assert service.get_agent(agent.id).name == "Researcher"


class TestRemoveTools:
    async def test_a_tool_is_detached(self, service):
        created = await service.create_agent(a_request(tools=["t1", "t2"]))
        assert await service.remove_tools(created.id, ["t1"]) is True
        assert service.get_agent(created.id).tools == ["t2"]

    async def test_detaching_produces_a_new_version(self, service):
        created = await service.create_agent(a_request(tools=["t1"]))
        await service.remove_tools(created.id, ["t1"])
        assert service.get_agent(created.id).version == 2

    async def test_detaching_a_tool_the_agent_never_had_is_a_no_op(
        self, service, agent
    ):
        assert await service.remove_tools(agent.id, ["nope"]) is True
        assert service.get_agent(agent.id).tools == []

    async def test_every_named_tool_is_removed(self, service):
        created = await service.create_agent(a_request(tools=["t1", "t2", "t3"]))
        await service.remove_tools(created.id, ["t1", "t3"])
        assert service.get_agent(created.id).tools == ["t2"]

    async def test_detaching_from_a_missing_agent_reports_failure(self, service):
        assert await service.remove_tools(str(ObjectId()), ["t1"]) is False

    async def test_the_earlier_version_keeps_the_tool(self, service):
        created = await service.create_agent(a_request(tools=["t1"]))
        await service.remove_tools(created.id, ["t1"])
        assert service.get_agent(f"{created.base_id}-v1").tools == ["t1"]


class TestGetAgentsByTool:
    async def test_no_agents_use_the_tool(self, service, agent):
        assert service.get_agents_by_tool("t1") == []

    async def test_an_agent_carrying_the_tool_is_found(self, service):
        created = await service.create_agent(a_request(tools=["t1"]))
        assert [a.id for a in service.get_agents_by_tool("t1")] == [created.id]

    async def test_only_the_latest_version_is_returned(self, service):
        created = await service.create_agent(a_request(tools=["t1"]))
        await service.update_agent(created.id, AgentUpdateRequest(name="Changed"))
        found = service.get_agents_by_tool("t1")
        assert len(found) == 1 and found[0].name == "Changed"

    async def test_an_agent_that_dropped_the_tool_is_not_returned(self, service):
        """Detaching writes a new version rather than rewriting the old one,
        so filtering before reducing to the head kept reporting the agent —
        and reported its stale definition while doing so."""
        created = await service.create_agent(a_request(tools=["t1"]))
        await service.remove_tools(created.id, ["t1"])
        assert service.get_agents_by_tool("t1") == []

    async def test_an_agent_that_later_gained_the_tool_is_returned(self, service):
        created = await service.create_agent(a_request())
        await service.add_tools(created.id, ["t1"])
        assert [a.id for a in service.get_agents_by_tool("t1")] == [created.id]

    async def test_several_agents_can_share_a_tool(self, service):
        await service.create_agent(a_request(tools=["t1"]))
        await service.create_agent(a_request(tools=["t1"]))
        assert len(service.get_agents_by_tool("t1")) == 2

    async def test_another_organizations_agents_are_not_returned(
        self, llm_service, agents_collection, service
    ):
        await service.create_agent(a_request(tools=["t1"]))
        intruder = AgentService(
            llm_service, TenantCollection(agents_collection, OTHER_ORG)
        )
        assert intruder.get_agents_by_tool("t1") == []


# --- tool-name resolution -------------------------------------------------


class TestResolveToolNames:
    def test_delegate_task_is_always_available(self, service):
        """Delegation is wired in via a toolset rather than the tools list, so
        the prompt has to be told about it unconditionally."""
        assert service._resolve_tool_names([]) == ["delegate_task"]

    def test_it_is_not_added_twice(self, service):
        assert (
            service._resolve_tool_names(["delegate_task"]).count("delegate_task") == 1
        )

    def test_ids_pass_through_when_there_is_no_tools_collection(self, service):
        assert "web_search" in service._resolve_tool_names(["web_search"])

    def test_an_id_is_resolved_to_its_registered_name(
        self, service_with_tools, tools_collection
    ):
        tool_id = tools_collection.insert_one(
            {"name": "web_search", "org_id": ORG_ID}
        ).inserted_id
        assert "web_search" in service_with_tools._resolve_tool_names([str(tool_id)])

    def test_a_name_is_accepted_where_an_id_is_expected(
        self, service_with_tools, tools_collection
    ):
        """Older agents stored tool names directly, so the lookup falls back to
        a name match rather than dropping the tool."""
        tools_collection.insert_one({"name": "web_search", "org_id": ORG_ID})
        assert "web_search" in service_with_tools._resolve_tool_names(["web_search"])

    def test_an_unknown_id_contributes_nothing(self, service_with_tools):
        assert service_with_tools._resolve_tool_names([str(ObjectId())]) == [
            "delegate_task"
        ]

    def test_an_unresolvable_name_contributes_nothing(self, service_with_tools):
        assert service_with_tools._resolve_tool_names(["nope"]) == ["delegate_task"]

    def test_another_organizations_tool_does_not_resolve(
        self, llm_service, agents_collection, tools_collection
    ):
        tools_collection.insert_one({"name": "web_search", "org_id": OTHER_ORG})
        service = AgentService(
            llm_service,
            TenantCollection(agents_collection, ORG_ID),
            TenantCollection(tools_collection, ORG_ID),
        )
        assert service._resolve_tool_names(["web_search"]) == ["delegate_task"]

    async def test_resolved_names_reach_the_system_prompt(
        self, service_with_tools, tools_collection
    ):
        tools_collection.insert_one({"name": "web_search", "org_id": ORG_ID})
        created = await service_with_tools.create_agent(a_request(tools=["web_search"]))
        assert "web_search" in created.system_prompt


# --- runnable agent -------------------------------------------------------


class TestGetRunnableAgent:
    async def test_a_runnable_agent_is_built(self, service, agent):
        assert service.get_runnable_agent(agent.id) is not None

    async def test_a_missing_agent_raises(self, service):
        from app.exceptions import AgentNotFoundError

        with pytest.raises(AgentNotFoundError):
            service.get_runnable_agent(str(ObjectId()))

    async def test_a_missing_llm_raises(self, service, llm_service, agent):
        """An agent pointing at a deleted LLM is a configuration fault, not a
        missing agent — the two produce different status codes upstream."""
        from app.exceptions import SystemConfigurationError

        llm_service.get_llm.return_value = None
        with pytest.raises(SystemConfigurationError):
            service.get_runnable_agent(agent.id)

    async def test_the_agents_own_llm_is_used(self, service, llm_service, agent):
        service.get_runnable_agent(agent.id)
        llm_service.get_llm.assert_called_with("llm_1")

    async def test_the_system_prompt_is_attached(self, service, agent):
        runnable = service.get_runnable_agent(agent.id)
        assert "Researcher" in str(runnable._instructions)

    async def test_a_pinned_version_builds_its_own_definition(self, service, agent):
        await service.update_agent(agent.id, AgentUpdateRequest(name="Changed"))
        runnable = service.get_runnable_agent(f"{agent.base_id}-v1")
        assert "Researcher" in str(runnable._instructions)


class TestAgentLanguagePersistence:
    """Language is stored on the agent document, and every write path in this
    service rebuilds that document by hand. A path that forgets the field
    silently resets the agent to English — the same class of bug as a dropped
    version number, and just as invisible to the user who set it."""

    async def test_it_defaults_to_english(self, service, agent):
        assert agent.language is AgentLanguage.ENGLISH

    async def test_it_is_stored_as_a_plain_string(self, service, agents_collection):
        """Anything reading this collection that is not the Pydantic model —
        a migration, a shell, an aggregation — should see a bare value."""
        await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        assert agents_collection.find_one({})["language"] == "filipino"

    async def test_it_round_trips_through_creation(self, service):
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        assert created.language is AgentLanguage.FILIPINO

    async def test_a_filipino_agent_gets_the_taglish_prompt_on_creation(self, service):
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        assert "Taglish" in created.system_prompt

    async def test_the_prompt_is_rebuilt_with_the_language_on_read(self, service):
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        assert "Taglish" in service.get_agent(created.id).system_prompt

    async def test_an_english_agent_gets_no_taglish_prompt(self, service, agent):
        assert "Taglish" not in service.get_agent(agent.id).system_prompt

    async def test_it_survives_an_unrelated_versioned_edit(self, service):
        """Renaming a Filipino agent must not quietly turn it English."""
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        updated = await service.update_agent(
            created.id, AgentUpdateRequest(name="Renamed")
        )
        assert updated.language is AgentLanguage.FILIPINO

    async def test_it_survives_an_unrelated_in_place_edit(self, service):
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        updated = await service.update_agent(
            created.id, AgentUpdateRequest(name="Renamed"), new_version=False
        )
        assert updated.language is AgentLanguage.FILIPINO

    async def test_it_can_be_switched_back_to_english(self, service):
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        updated = await service.update_agent(
            created.id, AgentUpdateRequest(language=AgentLanguage.ENGLISH)
        )
        assert updated.language is AgentLanguage.ENGLISH
        assert "Taglish" not in updated.system_prompt

    async def test_switching_language_rebuilds_the_prompt(self, service, agent):
        updated = await service.update_agent(
            agent.id, AgentUpdateRequest(language=AgentLanguage.FILIPINO)
        )
        assert "Taglish" in updated.system_prompt

    async def test_it_survives_attaching_a_tool(self, service):
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        await service.add_tools(created.id, ["web_search"])
        assert service.get_agent(created.id).language is AgentLanguage.FILIPINO

    async def test_it_survives_detaching_a_tool(self, service):
        created = await service.create_agent(
            a_request(language=AgentLanguage.FILIPINO, tools=["web_search"])
        )
        await service.remove_tools(created.id, ["web_search"])
        assert service.get_agent(created.id).language is AgentLanguage.FILIPINO

    async def test_an_agent_predating_the_field_still_loads(
        self, service, agents_collection
    ):
        """Existing documents carry no `language` key and are not migrated."""
        created = await service.create_agent(a_request())
        agents_collection.update_one({}, {"$unset": {"language": ""}})
        assert service.get_agent(created.id).language is AgentLanguage.ENGLISH

    async def test_a_pinned_version_keeps_the_language_it_was_created_with(
        self, service
    ):
        """Pinning exists so a validated definition cannot shift underneath a
        caller; the language is part of that definition."""
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        await service.update_agent(
            created.id, AgentUpdateRequest(language=AgentLanguage.ENGLISH)
        )
        assert (
            service.get_agent(f"{created.base_id}-v1").language
            is AgentLanguage.FILIPINO
        )

    async def test_the_runnable_agent_carries_the_taglish_instruction(self, service):
        """The prompt on the response is informational; this is the one that
        actually reaches the model."""
        created = await service.create_agent(a_request(language=AgentLanguage.FILIPINO))
        runnable = service.get_runnable_agent(created.id)
        assert "Taglish" in str(runnable._instructions)
