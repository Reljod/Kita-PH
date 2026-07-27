"""Tests for app.services.tool_service.

Registration is what makes a tool attachable to an agent, and it only accepts
names the discovery registry actually knows -- otherwise an agent could be
built around a tool that will never resolve at run time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.db import TenantCollection
from app.exceptions import (
    SystemConfigurationError,
    ToolNotFoundError,
    ToolRegistrationError,
)
from app.services.tool_service import ToolService

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"
A_REAL_TOOL = "search_memory"


@pytest.fixture
def tools_collection(mongo_db):
    return mongo_db["tools"]


@pytest.fixture
def web_search_service() -> MagicMock:
    service = MagicMock(name="web_search_service")
    service.search = AsyncMock(return_value={"organic": []})
    return service


@pytest.fixture
def service(web_search_service, tools_collection) -> ToolService:
    return ToolService(web_search_service, TenantCollection(tools_collection, ORG_ID))


@pytest.fixture
def unbacked(web_search_service) -> ToolService:
    """The web-search path is used without a collection (see tools/web_search)."""
    return ToolService(web_search_service)


# --- web search delegation ------------------------------------------------


class TestWebSearch:
    async def test_the_search_is_delegated(self, service, web_search_service):
        assert await service.web_search("kita") == {"organic": []}
        web_search_service.search.assert_awaited_once()

    async def test_the_options_are_forwarded(self, service, web_search_service):
        await service.web_search(
            "kita", country="ph", language="tl", page=3, search_type="news"
        )
        kwargs = web_search_service.search.await_args.kwargs
        assert kwargs["country"] == "ph"
        assert kwargs["language"] == "tl"
        assert kwargs["page"] == 3
        assert kwargs["search_type"] == "news"

    async def test_it_works_without_a_collection(self, unbacked):
        """Nothing about a web search touches the tools registry."""
        assert await unbacked.web_search("kita") == {"organic": []}

    async def test_a_provider_failure_propagates(self, service, web_search_service):
        web_search_service.search = AsyncMock(side_effect=RuntimeError("serper down"))
        with pytest.raises(RuntimeError):
            await service.web_search("kita")


# --- registration ---------------------------------------------------------


class TestRegisterTool:
    async def test_a_known_tool_is_registered(self, service, tools_collection):
        assert await service.register_tool(A_REAL_TOOL) is True
        assert tools_collection.count_documents({"name": A_REAL_TOOL}) == 1

    async def test_the_description_comes_from_the_registry(
        self, service, tools_collection
    ):
        """It is what the agent's prompt advertises, so it has to track the
        docstring rather than being typed in twice."""
        await service.register_tool(A_REAL_TOOL)
        assert tools_collection.find_one({})["description"].strip()

    async def test_it_is_scoped_to_the_organization(self, service, tools_collection):
        await service.register_tool(A_REAL_TOOL)
        assert tools_collection.find_one({})["org_id"] == ORG_ID

    async def test_an_unknown_tool_is_refused(self, service, tools_collection):
        """Registering one would let an agent be built around a tool that can
        never resolve at run time."""
        with pytest.raises(ToolRegistrationError):
            await service.register_tool("not_a_real_tool")
        assert tools_collection.count_documents({}) == 0

    async def test_registering_twice_is_idempotent(self, service, tools_collection):
        """Scaffolding re-runs on every organization, and a duplicate row
        would show the tool twice in the picker."""
        await service.register_tool(A_REAL_TOOL)
        assert await service.register_tool(A_REAL_TOOL) is True
        assert tools_collection.count_documents({}) == 1

    async def test_two_organizations_register_independently(
        self, web_search_service, tools_collection
    ):
        await ToolService(
            web_search_service, TenantCollection(tools_collection, ORG_ID)
        ).register_tool(A_REAL_TOOL)
        await ToolService(
            web_search_service, TenantCollection(tools_collection, OTHER_ORG)
        ).register_tool(A_REAL_TOOL)
        assert tools_collection.count_documents({}) == 2

    async def test_without_a_collection_it_is_a_configuration_error(self, unbacked):
        with pytest.raises(SystemConfigurationError):
            await unbacked.register_tool(A_REAL_TOOL)


class TestDeregisterTool:
    async def test_a_tool_is_deregistered(self, service, tools_collection):
        await service.register_tool(A_REAL_TOOL)
        tool_id = str(tools_collection.find_one({})["_id"])
        assert await service.deregister_tool(tool_id) is True
        assert tools_collection.count_documents({}) == 0

    async def test_a_missing_tool_raises(self, service):
        with pytest.raises(ToolNotFoundError):
            await service.deregister_tool(str(ObjectId()))

    async def test_a_malformed_id_raises(self, service):
        with pytest.raises(ToolRegistrationError):
            await service.deregister_tool("nope")

    async def test_another_organization_cannot_deregister_it(
        self, service, web_search_service, tools_collection
    ):
        await service.register_tool(A_REAL_TOOL)
        tool_id = str(tools_collection.find_one({})["_id"])
        intruder = ToolService(
            web_search_service, TenantCollection(tools_collection, OTHER_ORG)
        )
        with pytest.raises(ToolNotFoundError):
            await intruder.deregister_tool(tool_id)
        assert tools_collection.count_documents({}) == 1

    async def test_without_a_collection_it_is_a_configuration_error(self, unbacked):
        with pytest.raises(SystemConfigurationError):
            await unbacked.deregister_tool(str(ObjectId()))


# --- reads ----------------------------------------------------------------


class TestGetTools:
    async def test_no_tools_yields_an_empty_list(self, service):
        assert await service.get_tools() == []

    async def test_the_registered_tools_are_listed(self, service):
        await service.register_tool(A_REAL_TOOL)
        assert len(await service.get_tools()) == 1

    async def test_another_organizations_tools_are_not_listed(
        self, service, web_search_service, tools_collection
    ):
        await service.register_tool(A_REAL_TOOL)
        intruder = ToolService(
            web_search_service, TenantCollection(tools_collection, OTHER_ORG)
        )
        assert await intruder.get_tools() == []

    async def test_without_a_collection_it_is_a_configuration_error(self, unbacked):
        with pytest.raises(SystemConfigurationError):
            await unbacked.get_tools()


class TestGetTool:
    async def test_a_tool_is_returned(self, service, tools_collection):
        await service.register_tool(A_REAL_TOOL)
        tool_id = str(tools_collection.find_one({})["_id"])
        assert (await service.get_tool(tool_id))["name"] == A_REAL_TOOL

    async def test_a_missing_tool_raises(self, service):
        with pytest.raises(ToolNotFoundError):
            await service.get_tool(str(ObjectId()))

    async def test_a_malformed_id_raises_not_found(self, service):
        """The id comes from the URL, so a bad one is a 404 rather than a
        bson error surfacing as a 500."""
        with pytest.raises(ToolNotFoundError):
            await service.get_tool("nope")

    async def test_another_organization_cannot_read_it(
        self, service, web_search_service, tools_collection
    ):
        await service.register_tool(A_REAL_TOOL)
        tool_id = str(tools_collection.find_one({})["_id"])
        intruder = ToolService(
            web_search_service, TenantCollection(tools_collection, OTHER_ORG)
        )
        with pytest.raises(ToolNotFoundError):
            await intruder.get_tool(tool_id)

    async def test_without_a_collection_it_is_a_configuration_error(self, unbacked):
        with pytest.raises(SystemConfigurationError):
            await unbacked.get_tool(str(ObjectId()))


class TestGetToolByName:
    async def test_a_tool_is_returned(self, service):
        await service.register_tool(A_REAL_TOOL)
        assert (await service.get_tool_by_name(A_REAL_TOOL))["name"] == A_REAL_TOOL

    async def test_an_unregistered_name_raises(self, service):
        """Agent scaffolding resolves tool ids by name; the caller has to be
        able to tell "not registered here" from "registered with no id"."""
        with pytest.raises(ToolNotFoundError):
            await service.get_tool_by_name(A_REAL_TOOL)

    async def test_another_organizations_tool_is_not_visible(
        self, service, web_search_service, tools_collection
    ):
        await service.register_tool(A_REAL_TOOL)
        intruder = ToolService(
            web_search_service, TenantCollection(tools_collection, OTHER_ORG)
        )
        with pytest.raises(ToolNotFoundError):
            await intruder.get_tool_by_name(A_REAL_TOOL)

    async def test_without_a_collection_it_is_a_configuration_error(self, unbacked):
        with pytest.raises(SystemConfigurationError):
            await unbacked.get_tool_by_name(A_REAL_TOOL)
