"""Tests for app.services.tools — the functions agents actually call.

Every tool here returns a *string* on failure rather than raising, because the
return value goes back into the model's context as a tool result. An exception
escaping would abort the whole run; a message lets the agent recover or say
what went wrong. That contract is what most of these assert.

The registry functions at the bottom drive tool discovery, and their failure
mode is silence: a tool that stops being discovered simply disappears from
every agent's prompt.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.tools import (
    get_available_tools,
    get_tools_by_names,
    get_toolsets_by_names,
)
from app.services.tools.agent_tools import get_available_agents
from app.services.tools.delegation_tools import delegate_task
from app.services.tools.file_tools import resolve_file_id
from app.services.tools.memory_tools import rag_search, search_memory
from app.services.tools.web_search import web_search

ORG_ID = "org_test_0001"
AGENT_ID = "agent_1"
STATUS_KEY = "status-1"


def a_context(**deps) -> SimpleNamespace:
    """A stand-in RunContext; the tools only read `deps` and `usage`."""
    base = {"org_id": ORG_ID, "agent_id": AGENT_ID}
    base.update(deps)
    return SimpleNamespace(deps=base, usage=MagicMock())


def a_result(title="Title", content="Content", question=None, answer=None):
    return SimpleNamespace(
        title=title, content=content, question=question, answer=answer
    )


@pytest.fixture
def status_service() -> MagicMock:
    service = MagicMock(name="agent_status_service")
    service.update_step = AsyncMock()
    return service


@pytest.fixture
def registry(monkeypatch, status_service) -> MagicMock:
    services = MagicMock(name="registry")
    services.agent_status_service = status_service
    monkeypatch.setattr(
        "app.dependencies.services.get_services", lambda org_id: services
    )
    return services


# --- memory ---------------------------------------------------------------


class TestSearchMemory:
    async def test_results_are_formatted_for_the_model(self, registry):
        service = MagicMock()
        service.search = AsyncMock(return_value=[a_result(title="Runbook")])
        result = await search_memory(a_context(rag_service=service), "deploy")
        assert "Runbook" in result

    async def test_an_enriched_memory_is_rendered_as_a_question(self, registry):
        """Enrichment turns a note into a Q/A pair; showing the raw content
        instead would discard the work."""
        service = MagicMock()
        service.search = AsyncMock(
            return_value=[a_result(question="How do I deploy?", answer="Run make")]
        )
        result = await search_memory(a_context(rag_service=service), "deploy")
        assert "How do I deploy?" in result and "Run make" in result

    async def test_no_results_says_so_rather_than_returning_nothing(self, registry):
        """An empty string reads to the model as a broken tool; a sentence
        tells it to try another approach."""
        service = MagicMock()
        service.search = AsyncMock(return_value=[])
        result = await search_memory(a_context(rag_service=service), "deploy")
        assert "No relevant information" in result

    async def test_the_search_is_scoped_to_the_calling_agent(self, registry):
        service = MagicMock()
        service.search = AsyncMock(return_value=[])
        await search_memory(a_context(rag_service=service), "deploy")
        assert service.search.await_args.kwargs["agent_id"] == AGENT_ID

    async def test_the_limit_is_forwarded(self, registry):
        service = MagicMock()
        service.search = AsyncMock(return_value=[])
        await search_memory(a_context(rag_service=service), "deploy", 3)
        assert service.search.await_args.kwargs["limit"] == 3

    async def test_the_retrieval_step_is_published(self, registry, status_service):
        service = MagicMock()
        service.search = AsyncMock(return_value=[])
        await search_memory(
            a_context(rag_service=service, status_key=STATUS_KEY), "deploy"
        )
        assert status_service.update_step.await_args[0][1] == "retrieve_facts_vector"

    async def test_a_status_failure_does_not_stop_the_search(
        self, registry, status_service
    ):
        """Status is telemetry; a slow Redis must not cost the agent its
        answer -- hence the timeout around it."""
        status_service.update_step = AsyncMock(side_effect=RuntimeError("redis down"))
        service = MagicMock()
        service.search = AsyncMock(return_value=[a_result()])
        result = await search_memory(
            a_context(rag_service=service, status_key=STATUS_KEY), "deploy"
        )
        assert "Title" in result


class TestRagSearch:
    async def test_results_are_formatted_for_the_model(self, registry):
        service = MagicMock()
        service.search = AsyncMock(return_value=[a_result(title="Handbook")])
        result = await rag_search(a_context(retrieval_service=service), "benefits")
        assert "Handbook" in result

    async def test_no_results_says_so(self, registry):
        service = MagicMock()
        service.search = AsyncMock(return_value=[])
        result = await rag_search(a_context(retrieval_service=service), "benefits")
        assert "No relevant information" in result

    async def test_it_falls_back_to_plain_memory_search(self, registry):
        """The hybrid retriever is only wired up for organizations with
        ingested documents; without it the agent still has its memories."""
        rag = MagicMock()
        rag.search = AsyncMock(return_value=[a_result(title="From memory")])
        result = await rag_search(a_context(rag_service=rag), "benefits")
        assert "From memory" in result

    async def test_the_retrieval_step_is_published(self, registry, status_service):
        service = MagicMock()
        service.search = AsyncMock(return_value=[])
        await rag_search(
            a_context(retrieval_service=service, status_key=STATUS_KEY), "benefits"
        )
        status_service.update_step.assert_awaited()


# --- delegation -----------------------------------------------------------


def a_sub_agent(output):
    agent = MagicMock()
    agent.run = AsyncMock(return_value=SimpleNamespace(output=output))
    return agent


class TestDelegateTask:
    async def test_the_sub_agents_answer_is_returned(self, registry):
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent("the answer")
        result = await delegate_task(a_context(agent_service=service), "research it")
        assert result == "the answer"

    async def test_a_missing_service_is_reported_not_raised(self, registry):
        assert "not found" in await delegate_task(a_context(), "research it")

    async def test_the_task_reaches_the_sub_agent(self, registry):
        sub = a_sub_agent("done")
        service = MagicMock()
        service.get_runnable_agent.return_value = sub
        await delegate_task(a_context(agent_service=service), "research it")
        assert sub.run.await_args[0][0] == "research it"

    async def test_an_explicit_target_agent_is_used(self, registry):
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent("done")
        await delegate_task(
            a_context(agent_service=service), "research it", target_agent_id="agent_2"
        )
        assert service.get_runnable_agent.call_args.kwargs["agent_id"] == "agent_2"

    async def test_it_defaults_to_the_calling_agent(self, registry):
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent("done")
        await delegate_task(a_context(agent_service=service), "research it")
        assert service.get_runnable_agent.call_args.kwargs["agent_id"] == AGENT_ID

    async def test_the_dependencies_are_passed_down(self, registry):
        """A sub-agent without deps has no tools, so it can only answer from
        parametric knowledge."""
        sub = a_sub_agent("done")
        service = MagicMock()
        service.get_runnable_agent.return_value = sub
        ctx = a_context(agent_service=service)
        await delegate_task(ctx, "research it")
        assert sub.run.await_args.kwargs["deps"] is ctx.deps

    async def test_the_token_usage_is_shared(self, registry):
        """Otherwise a delegating agent's spend is invisible in the parent
        run's accounting."""
        sub = a_sub_agent("done")
        service = MagicMock()
        service.get_runnable_agent.return_value = sub
        ctx = a_context(agent_service=service)
        await delegate_task(ctx, "research it")
        assert sub.run.await_args.kwargs["usage"] is ctx.usage

    async def test_json_output_is_parsed_into_structure(self, registry):
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent('{"total": 42}')
        assert await delegate_task(a_context(agent_service=service), "count") == {
            "total": 42
        }

    async def test_a_fenced_json_block_is_unwrapped(self, registry):
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent(
            '```json\n{"total": 42}\n```'
        )
        assert await delegate_task(a_context(agent_service=service), "count") == (
            {"total": 42}
        )

    async def test_non_json_output_stays_a_string(self, registry):
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent("just prose")
        assert await delegate_task(a_context(agent_service=service), "x") == (
            "just prose"
        )

    async def test_a_very_large_json_answer_stays_a_string(self, registry):
        """Handing a huge parsed structure back would blow the parent's
        context window in one step."""
        payload = '{"data": "' + "x" * 30000 + '"}'
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent(payload)
        assert isinstance(
            await delegate_task(a_context(agent_service=service), "x"), str
        )

    async def test_a_failure_is_reported_not_raised(self, registry):
        """An exception here aborts the parent run; a message lets it try
        something else."""
        service = MagicMock()
        service.get_runnable_agent.side_effect = RuntimeError("no such agent")
        result = await delegate_task(a_context(agent_service=service), "x")
        assert "Error during delegation" in result

    async def test_the_delegation_step_is_published(self, registry, status_service):
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent("done")
        await delegate_task(
            a_context(agent_service=service, status_key=STATUS_KEY), "x"
        )
        assert status_service.update_step.await_args[0][1] == "delegated_task"

    async def test_a_status_failure_does_not_stop_the_delegation(
        self, registry, status_service
    ):
        status_service.update_step = AsyncMock(side_effect=RuntimeError("redis down"))
        service = MagicMock()
        service.get_runnable_agent.return_value = a_sub_agent("done")
        assert (
            await delegate_task(
                a_context(agent_service=service, status_key=STATUS_KEY), "x"
            )
            == "done"
        )


# --- agent discovery ------------------------------------------------------


class TestGetAvailableAgents:
    async def test_the_agents_are_listed(self, registry):
        service = MagicMock()
        service.get_all_agents.return_value = [
            SimpleNamespace(id="a2", name="Scribe", role="writer", goal="write")
        ]
        listed = await get_available_agents(a_context(agent_service=service))
        assert listed[0]["name"] == "Scribe"

    async def test_the_calling_agent_is_excluded(self, registry):
        """Delegating to yourself is an infinite loop that only stops when
        the run budget runs out."""
        service = MagicMock()
        service.get_all_agents.return_value = [
            SimpleNamespace(id=AGENT_ID, name="Me", role="r", goal="g"),
            SimpleNamespace(id="a2", name="Other", role="r", goal="g"),
        ]
        listed = await get_available_agents(a_context(agent_service=service))
        assert [a["name"] for a in listed] == ["Other"]

    async def test_a_missing_service_is_reported_not_raised(self, registry):
        assert "error" in (await get_available_agents(a_context()))[0]

    async def test_a_failure_is_reported_not_raised(self, registry):
        service = MagicMock()
        service.get_all_agents.side_effect = RuntimeError("mongo down")
        assert (
            "error" in (await get_available_agents(a_context(agent_service=service)))[0]
        )


# --- files ----------------------------------------------------------------


class TestResolveFileId:
    async def test_the_id_is_extracted_from_the_path(self, registry):
        service = MagicMock()
        service.get_file = AsyncMock(return_value=SimpleNamespace(id="abc"))
        assert await resolve_file_id(a_context(file_service=service), "abc.pdf") == (
            "abc"
        )

    async def test_a_path_without_an_extension_is_accepted(self, registry):
        service = MagicMock()
        service.get_file = AsyncMock(return_value=SimpleNamespace(id="abc"))
        assert await resolve_file_id(a_context(file_service=service), "abc") == "abc"

    async def test_a_missing_service_is_reported_not_raised(self, registry):
        assert "Error" in await resolve_file_id(a_context(), "abc.pdf")

    async def test_an_unknown_file_is_reported_not_raised(self, registry):
        service = MagicMock()
        service.get_file = AsyncMock(side_effect=RuntimeError("not found"))
        assert "Error" in await resolve_file_id(
            a_context(file_service=service), "x.pdf"
        )

    async def test_a_falsy_lookup_is_reported(self, registry):
        service = MagicMock()
        service.get_file = AsyncMock(return_value=None)
        assert "not found" in await resolve_file_id(
            a_context(file_service=service), "x.pdf"
        )


# --- web search -----------------------------------------------------------


class TestWebSearch:
    @pytest.fixture
    def tool_service(self, monkeypatch) -> MagicMock:
        service = MagicMock(name="tool_service")
        service.web_search = AsyncMock(return_value={"organic": []})
        monkeypatch.setattr(
            "app.services.tools.web_search.ToolService", lambda *a, **k: service
        )
        monkeypatch.setattr(
            "app.services.tools.web_search.SerperSearchService", MagicMock()
        )
        return service

    async def test_results_are_formatted_for_the_model(self, registry, tool_service):
        tool_service.web_search = AsyncMock(
            return_value={
                "organic": [
                    {"title": "T", "link": "https://x", "snippet": "S"},
                ]
            }
        )
        result = await web_search(a_context(), "kita")
        assert "T" in result and "https://x" in result and "S" in result

    async def test_no_results_says_so(self, registry, tool_service):
        assert "No relevant information" in await web_search(a_context(), "kita")

    async def test_a_malformed_response_says_so(self, registry, tool_service):
        tool_service.web_search = AsyncMock(return_value={"error": "quota exceeded"})
        assert "No relevant information" in await web_search(a_context(), "kita")

    async def test_results_are_capped(self, registry, tool_service):
        """The whole block lands in the model's context; an uncapped page of
        results would crowd out the conversation."""
        tool_service.web_search = AsyncMock(
            return_value={"organic": [{"title": f"T{i}"} for i in range(20)]}
        )
        assert (await web_search(a_context(), "kita")).count("Link:") == 8

    async def test_a_provider_failure_is_reported_not_raised(
        self, registry, tool_service
    ):
        tool_service.web_search = AsyncMock(side_effect=RuntimeError("serper down"))
        assert "Error performing web search" in await web_search(a_context(), "kita")

    async def test_the_search_options_are_forwarded(self, registry, tool_service):
        await web_search(a_context(), "kita", country="ph", search_type="news")
        assert tool_service.web_search.await_args.kwargs["country"] == "ph"
        assert tool_service.web_search.await_args.kwargs["search_type"] == "news"

    async def test_the_web_step_is_published(
        self, registry, tool_service, status_service
    ):
        await web_search(a_context(status_key=STATUS_KEY), "kita")
        assert status_service.update_step.await_args[0][1] == "retrieve_facts_web"


# --- the registry ---------------------------------------------------------


class TestToolRegistry:
    def test_the_known_tools_are_discovered(self):
        """Discovery drives what every agent's prompt advertises, so a tool
        that stops being found simply disappears."""
        available = get_available_tools()
        for name in ("search_memory", "rag_search", "web_search", "delegate_task"):
            assert name in available

    def test_each_tool_carries_a_description(self):
        """The description is the only thing telling the model when to reach
        for the tool."""
        assert all(desc.strip() for desc in get_available_tools().values())

    def test_a_named_tool_is_returned(self):
        assert len(get_tools_by_names(["search_memory"])) == 1

    def test_unknown_names_return_nothing(self):
        assert get_tools_by_names(["not_a_tool"]) == []

    def test_no_names_return_nothing(self):
        assert get_tools_by_names([]) == []

    def test_a_repeated_name_is_returned_once(self):
        assert len(get_tools_by_names(["search_memory", "search_memory"])) == 1

    def test_tools_from_several_modules_are_collected(self):
        found = get_tools_by_names(["search_memory", "delegate_task"])
        assert len(found) == 2

    def test_a_toolset_is_returned_for_a_named_tool(self):
        assert len(get_toolsets_by_names(["delegate_task"])) == 1

    def test_unknown_names_return_no_toolsets(self):
        assert get_toolsets_by_names(["not_a_tool"]) == []

    def test_a_toolset_is_returned_once_for_two_of_its_tools(self):
        """search_memory and rag_search share the memory toolset; returning
        it twice would register every tool in it twice."""
        assert len(get_toolsets_by_names(["search_memory", "rag_search"])) == 1
