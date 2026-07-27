"""Tests for app.services.adaptive_rag_service.

This is the Adaptive-RAG / Self-RAG loop: route the query, retrieve, generate,
then grade the draft for groundedness and completeness and go round again if it
fails. Every grader calls a cheap LLM, so the behaviour worth pinning is what
happens when those calls return something unexpected -- the fallbacks decide
whether a flaky grader degrades the answer or blocks it entirely.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.adaptive_rag_service import AdaptiveRagService

ORG_ID = "org_test_0001"
AGENT_ID = "agent_1"
STATUS_KEY = "status-1"


@pytest.fixture
def retrieval_service() -> MagicMock:
    service = MagicMock(name="retrieval_service")
    service.search = AsyncMock(return_value=[])
    return service


@pytest.fixture
def web_search_service() -> MagicMock:
    service = MagicMock(name="web_search_service")
    service.search = AsyncMock(return_value={})
    return service


@pytest.fixture
def status_service() -> MagicMock:
    service = MagicMock(name="agent_status_service")
    service.update_step = AsyncMock()
    return service


@pytest.fixture
def registry(monkeypatch, status_service) -> MagicMock:
    """`get_services` is imported inside each method, so patch it at source."""
    services = MagicMock(name="registry")
    services.agent_status_service = status_service
    services.llm_service.run = AsyncMock(return_value="")
    monkeypatch.setattr(
        "app.dependencies.services.get_services", lambda org_id: services
    )
    return services


@pytest.fixture
def service(retrieval_service, web_search_service, registry) -> AdaptiveRagService:
    return AdaptiveRagService(ORG_ID, retrieval_service, web_search_service)


def llm_returning(*responses):
    """A _call_cheap_llm replacement that replays the given answers in turn."""
    queue = list(responses)

    async def _call(*args, **kwargs):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return _call


# --- chat history formatting ----------------------------------------------


class TestFormatChatHistory:
    def test_no_history_is_empty(self, service):
        assert service._format_chat_history(None) == ""

    def test_an_empty_history_is_empty(self, service):
        assert service._format_chat_history([]) == ""

    def test_dict_messages_are_transcribed(self, service):
        history = [{"role": "user", "parts": [{"content": "hello"}]}]
        assert service._format_chat_history(history) == "User: hello"

    def test_the_agent_side_is_labelled(self, service):
        history = [{"role": "assistant", "parts": [{"content": "hi"}]}]
        assert service._format_chat_history(history) == "Agent: hi"

    def test_object_messages_are_transcribed(self, service):
        """pydantic-ai hands back objects, not dicts, on a live run, and the
        formatter dispatches on their class names."""

        class TextPart:
            def __init__(self, content):
                self.content = content

        class ModelRequest:
            parts = [TextPart("hello")]

        assert service._format_chat_history([ModelRequest()]) == "User: hello"

    def test_an_object_response_is_labelled_as_the_agent(self, service):
        class TextPart:
            def __init__(self, content):
                self.content = content

        class ModelResponse:
            parts = [TextPart("hi back")]

        assert service._format_chat_history([ModelResponse()]) == "Agent: hi back"

    def test_object_parts_of_other_kinds_are_skipped(self, service):
        class ToolCallPart:
            content = "should not appear"

        class ModelResponse:
            parts = [ToolCallPart()]

        assert service._format_chat_history([ModelResponse()]) == ""

    def test_parts_without_content_are_skipped(self, service):
        """Tool-call parts carry `args` rather than `content`."""
        history = [{"role": "user", "parts": [{"tool_name": "search"}]}]
        assert service._format_chat_history(history) == ""

    def test_only_the_last_turns_are_kept(self, service):
        """The transcript goes into a prompt, so an unbounded history would
        grow the condenser call without bound."""
        history = [{"role": "user", "parts": [{"content": f"m{i}"}]} for i in range(10)]
        assert service._format_chat_history(history, limit=2) == "User: m8\nUser: m9"


# --- condensing -----------------------------------------------------------


class TestCondenseQuery:
    async def test_no_history_returns_the_query_unchanged(self, service):
        """Nothing to resolve against, so spending an LLM call would be pure
        latency."""
        assert await service.condense_query("who is he?", None) == "who is he?"

    async def test_an_unusable_history_returns_the_query_unchanged(self, service):
        history = [{"role": "user", "parts": [{"tool_name": "search"}]}]
        assert await service.condense_query("who is he?", history) == "who is he?"

    async def test_the_condensed_query_is_returned(self, service, monkeypatch):
        history = [{"role": "user", "parts": [{"content": "tell me about Kita"}]}]
        monkeypatch.setattr(
            service, "_call_cheap_llm", llm_returning("who founded Kita?")
        )
        assert await service.condense_query("who?", history) == "who founded Kita?"

    async def test_quotes_are_stripped(self, service, monkeypatch):
        """Models like to wrap the rewrite in quotes; those would be searched
        for literally."""
        history = [{"role": "user", "parts": [{"content": "about Kita"}]}]
        monkeypatch.setattr(
            service, "_call_cheap_llm", llm_returning('"who founded Kita?"')
        )
        assert await service.condense_query("who?", history) == "who founded Kita?"

    async def test_a_failure_falls_back_to_the_original(self, service, monkeypatch):
        """Condensing is an optimisation; losing it must not lose the turn."""
        history = [{"role": "user", "parts": [{"content": "about Kita"}]}]
        monkeypatch.setattr(
            service, "_call_cheap_llm", AsyncMock(side_effect=RuntimeError("down"))
        )
        assert await service.condense_query("who?", history) == "who?"


# --- routing --------------------------------------------------------------


class TestRouteQuery:
    @pytest.mark.parametrize("choice", ["vector", "web", "none"])
    async def test_each_strategy_is_recognised(self, service, monkeypatch, choice):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning(choice))
        assert await service.route_query("q") == choice

    async def test_the_answer_is_normalised(self, service, monkeypatch):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning('  "WEB"  '))
        assert await service.route_query("q") == "web"

    async def test_a_wrapped_answer_is_still_understood(self, service, monkeypatch):
        """Small models pad their answers; a prose reply should not silently
        route everything to the default."""
        monkeypatch.setattr(
            service, "_call_cheap_llm", llm_returning("I would choose web here")
        )
        assert await service.route_query("q") == "web"

    async def test_an_unrecognisable_answer_defaults_to_internal_search(
        self, service, monkeypatch
    ):
        """Internal documents are the safer default: a web search on a private
        question leaks the query to a third party."""
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning("banana"))
        assert await service.route_query("q") == "vector"

    async def test_a_failure_defaults_to_internal_search(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "_call_cheap_llm", AsyncMock(side_effect=RuntimeError("down"))
        )
        assert await service.route_query("q") == "vector"


# --- graders --------------------------------------------------------------


class TestGradeRelevance:
    @pytest.mark.parametrize("answer,expected", [("YES", True), ("NO", False)])
    async def test_the_verdict_is_read(self, service, monkeypatch, answer, expected):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning(answer))
        assert await service.grade_relevance("q", "doc") is expected

    async def test_the_verdict_is_case_insensitive(self, service, monkeypatch):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning("yes"))
        assert await service.grade_relevance("q", "doc") is True

    async def test_a_failure_keeps_the_document(self, service, monkeypatch):
        """Dropping documents when the grader is down would quietly starve the
        generator of context."""
        monkeypatch.setattr(
            service, "_call_cheap_llm", AsyncMock(side_effect=RuntimeError("down"))
        )
        assert await service.grade_relevance("q", "doc") is True


class TestGradeGroundedness:
    async def test_no_facts_is_vacuously_grounded(self, service):
        assert await service.grade_groundedness([], "answer") is True

    @pytest.mark.parametrize("answer,expected", [("YES", True), ("NO", False)])
    async def test_the_verdict_is_read(self, service, monkeypatch, answer, expected):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning(answer))
        assert await service.grade_groundedness(["fact"], "answer") is expected

    async def test_a_failure_accepts_the_answer(self, service, monkeypatch):
        """Otherwise a grader outage turns every response into three retries
        and then the same draft anyway."""
        monkeypatch.setattr(
            service, "_call_cheap_llm", AsyncMock(side_effect=RuntimeError("down"))
        )
        assert await service.grade_groundedness(["fact"], "answer") is True


class TestGradeCompleteness:
    async def test_a_complete_verdict_is_read(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "_call_cheap_llm", llm_returning('{"is_complete": true}')
        )
        assert await service.grade_completeness("q", "a") is True

    async def test_an_incomplete_verdict_is_read(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "_call_cheap_llm", llm_returning('{"is_complete": false}')
        )
        assert await service.grade_completeness("q", "a") is False

    async def test_a_fenced_json_block_is_unwrapped(self, service, monkeypatch):
        """Models wrap JSON in markdown fences even when told not to."""
        monkeypatch.setattr(
            service,
            "_call_cheap_llm",
            llm_returning('```json\n{"is_complete": false}\n```'),
        )
        assert await service.grade_completeness("q", "a") is False

    async def test_a_missing_key_reads_as_incomplete(self, service, monkeypatch):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning("{}"))
        assert await service.grade_completeness("q", "a") is False

    async def test_unparseable_output_accepts_the_answer(self, service, monkeypatch):
        """The loop retries on incomplete, so treating a parse failure as
        incomplete would burn every attempt on a broken grader."""
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning("not json"))
        assert await service.grade_completeness("q", "a") is True

    async def test_a_failure_accepts_the_answer(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "_call_cheap_llm", AsyncMock(side_effect=RuntimeError("down"))
        )
        assert await service.grade_completeness("q", "a") is True


class TestRewriteQuery:
    async def test_the_rewrite_is_returned(self, service, monkeypatch):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning("kita founders"))
        assert await service.rewrite_query("who made it") == "kita founders"

    async def test_quotes_are_stripped(self, service, monkeypatch):
        monkeypatch.setattr(service, "_call_cheap_llm", llm_returning('"kita"'))
        assert await service.rewrite_query("q") == "kita"

    async def test_a_failure_falls_back_to_the_original(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "_call_cheap_llm", AsyncMock(side_effect=RuntimeError("down"))
        )
        assert await service.rewrite_query("original") == "original"


# --- retrieval ------------------------------------------------------------


class TestRetrieveFacts:
    @pytest.fixture(autouse=True)
    def keep_everything(self, service, monkeypatch):
        """Grade every document relevant unless a test says otherwise."""
        monkeypatch.setattr(service, "grade_relevance", AsyncMock(return_value=True))

    async def test_vector_search_results_become_facts(self, service, retrieval_service):
        retrieval_service.search = AsyncMock(
            return_value=[SimpleNamespace(content="internal fact")]
        )
        assert await service.retrieve_facts("vector", "q") == ["internal fact"]

    async def test_web_search_results_become_facts(self, service, web_search_service):
        web_search_service.search = AsyncMock(
            return_value={
                "organic": [
                    {"title": "T", "link": "https://x", "snippet": "S"},
                ]
            }
        )
        facts = await service.retrieve_facts("web", "q")
        assert "T" in facts[0] and "https://x" in facts[0] and "S" in facts[0]

    async def test_web_results_are_capped(self, service, web_search_service):
        """The whole set goes into the generator prompt; an uncapped page of
        results would blow the context window."""
        web_search_service.search = AsyncMock(
            return_value={"organic": [{"title": f"T{i}"} for i in range(20)]}
        )
        assert len(await service.retrieve_facts("web", "q")) == 8

    async def test_web_results_without_the_expected_shape_yield_nothing(
        self, service, web_search_service
    ):
        web_search_service.search = AsyncMock(return_value={"error": "quota"})
        assert await service.retrieve_facts("web", "q") == []

    async def test_an_unknown_strategy_retrieves_nothing(self, service):
        assert await service.retrieve_facts("telepathy", "q") == []

    async def test_a_vector_search_failure_yields_no_facts(
        self, service, retrieval_service
    ):
        """The generator can still answer from parametric knowledge; raising
        here would fail the whole turn."""
        retrieval_service.search = AsyncMock(side_effect=RuntimeError("index down"))
        assert await service.retrieve_facts("vector", "q") == []

    async def test_a_web_search_failure_yields_no_facts(
        self, service, web_search_service
    ):
        web_search_service.search = AsyncMock(side_effect=RuntimeError("serper down"))
        assert await service.retrieve_facts("web", "q") == []

    async def test_irrelevant_documents_are_filtered_out(
        self, service, retrieval_service, monkeypatch
    ):
        retrieval_service.search = AsyncMock(
            return_value=[
                SimpleNamespace(content="keep"),
                SimpleNamespace(content="drop"),
            ]
        )
        monkeypatch.setattr(
            service,
            "grade_relevance",
            AsyncMock(side_effect=lambda q, doc, **kw: doc == "keep"),
        )
        assert await service.retrieve_facts("vector", "q") == ["keep"]

    async def test_the_status_names_the_strategy(
        self, service, status_service, retrieval_service
    ):
        await service.retrieve_facts("vector", "q", status_key=STATUS_KEY)
        assert status_service.update_step.await_args[0][1] == "retrieve_facts_vector"

    async def test_no_status_is_published_without_a_key(self, service, status_service):
        await service.retrieve_facts("vector", "q")
        status_service.update_step.assert_not_awaited()

    async def test_a_status_failure_does_not_stop_retrieval(
        self, service, status_service, retrieval_service
    ):
        """Status is telemetry; it must never be why a search does not run."""
        status_service.update_step = AsyncMock(side_effect=RuntimeError("redis down"))
        retrieval_service.search = AsyncMock(
            return_value=[SimpleNamespace(content="fact")]
        )
        assert await service.retrieve_facts("vector", "q", status_key=STATUS_KEY) == [
            "fact"
        ]


# --- the full loop --------------------------------------------------------


def an_agent(response: str = "the answer") -> MagicMock:
    agent = MagicMock(name="agent")
    agent.run = AsyncMock(return_value=SimpleNamespace(data=response))
    return agent


class TestRunAgenticFlow:
    @pytest.fixture(autouse=True)
    def stub_graders(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "condense_query", AsyncMock(side_effect=lambda q, *a, **k: q)
        )
        monkeypatch.setattr(
            service, "rewrite_query", AsyncMock(side_effect=lambda q, **k: q)
        )
        monkeypatch.setattr(service, "grade_groundedness", AsyncMock(return_value=True))
        monkeypatch.setattr(service, "grade_completeness", AsyncMock(return_value=True))

    async def test_a_query_needing_no_retrieval_answers_directly(
        self, service, monkeypatch, retrieval_service
    ):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="none"))
        agent = an_agent()
        assert await service.run_agentic_flow("hello", agent, None) is (
            agent.run.return_value
        )
        retrieval_service.search.assert_not_awaited()

    async def test_retrieved_facts_reach_the_generator(self, service, monkeypatch):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        monkeypatch.setattr(
            service, "retrieve_facts", AsyncMock(return_value=["a known fact"])
        )
        agent = an_agent()
        await service.run_agentic_flow("q", agent, None)
        assert "a known fact" in agent.run.await_args.kwargs["instructions"]

    async def test_an_empty_internal_search_fails_over_to_the_web(
        self, service, monkeypatch
    ):
        """Answering "I don't know" when the web had the answer is the failure
        this failover exists to prevent."""
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        strategies = []

        async def retrieve(strategy, query, **kwargs):
            strategies.append(strategy)
            return [] if strategy == "vector" else ["web fact"]

        monkeypatch.setattr(service, "retrieve_facts", retrieve)
        await service.run_agentic_flow("q", an_agent(), None)
        assert strategies == ["vector", "web"]

    async def test_the_failover_does_not_revisit_a_strategy(self, service, monkeypatch):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        strategies = []

        async def retrieve(strategy, query, **kwargs):
            strategies.append(strategy)
            return []

        monkeypatch.setattr(service, "retrieve_facts", retrieve)
        await service.run_agentic_flow("q", an_agent(), None)
        assert strategies == ["vector", "web"]

    async def test_an_ungrounded_draft_is_retried(self, service, monkeypatch):
        """This is the hallucination gate: a draft the facts do not support
        must not be returned on the first pass."""
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        monkeypatch.setattr(service, "retrieve_facts", AsyncMock(return_value=["f"]))
        monkeypatch.setattr(
            service, "grade_groundedness", AsyncMock(side_effect=[False, True])
        )
        agent = an_agent()
        await service.run_agentic_flow("q", agent, None)
        assert agent.run.await_count == 2

    async def test_an_incomplete_draft_fails_over_to_the_other_strategy(
        self, service, monkeypatch
    ):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        strategies = []

        async def retrieve(strategy, query, **kwargs):
            strategies.append(strategy)
            return ["fact"]

        monkeypatch.setattr(service, "retrieve_facts", retrieve)
        monkeypatch.setattr(
            service, "grade_completeness", AsyncMock(side_effect=[False, True])
        )
        await service.run_agentic_flow("q", an_agent(), None)
        assert strategies == ["vector", "web"]

    async def test_an_exhausted_failover_rewrites_and_retries_instead(
        self, service, monkeypatch
    ):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        monkeypatch.setattr(service, "retrieve_facts", AsyncMock(return_value=["f"]))
        monkeypatch.setattr(
            service, "grade_completeness", AsyncMock(side_effect=[False, False, True])
        )
        rewrite = AsyncMock(side_effect=lambda q, **k: q)
        monkeypatch.setattr(service, "rewrite_query", rewrite)
        await service.run_agentic_flow("q", an_agent(), None)
        assert rewrite.await_count >= 2

    async def test_the_loop_gives_up_after_three_attempts(self, service, monkeypatch):
        """Without a ceiling a stubborn grader would loop against a paid model
        indefinitely."""
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        monkeypatch.setattr(service, "retrieve_facts", AsyncMock(return_value=["f"]))
        monkeypatch.setattr(
            service, "grade_completeness", AsyncMock(return_value=False)
        )
        agent = an_agent()
        assert await service.run_agentic_flow("q", agent, None) is not None
        assert agent.run.await_count == 3

    async def test_the_original_query_is_what_the_generator_answers(
        self, service, monkeypatch
    ):
        """Condensing and rewriting shape the *search*; answering the rewritten
        text would drift away from what the user actually asked."""
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="vector"))
        monkeypatch.setattr(service, "retrieve_facts", AsyncMock(return_value=["f"]))
        monkeypatch.setattr(
            service, "condense_query", AsyncMock(return_value="condensed form")
        )
        agent = an_agent()
        await service.run_agentic_flow("what about it?", agent, None)
        assert agent.run.await_args[0][0] == "what about it?"

    async def test_the_generation_step_is_published(
        self, service, monkeypatch, status_service
    ):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="none"))
        await service.run_agentic_flow("q", an_agent(), None, status_key=STATUS_KEY)
        steps = [call[0][1] for call in status_service.update_step.await_args_list]
        assert "generate_response" in steps

    async def test_a_status_failure_does_not_stop_generation(
        self, service, monkeypatch, status_service
    ):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="none"))
        status_service.update_step = AsyncMock(side_effect=RuntimeError("redis down"))
        assert await service.run_agentic_flow(
            "q", an_agent(), None, status_key=STATUS_KEY
        )

    async def test_the_tenant_reaches_the_generator_dependencies(
        self, service, monkeypatch
    ):
        monkeypatch.setattr(service, "route_query", AsyncMock(return_value="none"))
        agent = an_agent()
        await service.run_agentic_flow("q", agent, None, agent_id=AGENT_ID)
        deps = agent.run.await_args.kwargs["deps"]
        assert deps["org_id"] == ORG_ID and deps["agent_id"] == AGENT_ID


# --- the cheap LLM helper -------------------------------------------------


class TestCallCheapLlm:
    async def test_the_router_model_is_used(self, service, registry):
        await service._call_cheap_llm("sys", "user")
        assert registry.llm_service.run.await_args.kwargs["model_name"] == (
            service.model_name
        )

    async def test_the_prompts_are_passed_as_messages(self, service, registry):
        await service._call_cheap_llm("sys", "user")
        messages = registry.llm_service.run.await_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1] == {"role": "user", "content": "user"}

    async def test_grading_is_deterministic(self, service, registry):
        """A grader that changes its mind between identical calls makes the
        retry loop non-reproducible."""
        await service._call_cheap_llm("sys", "user")
        assert registry.llm_service.run.await_args.kwargs["temperature"] == 0.0

    async def test_json_mode_gets_a_larger_budget(self, service, registry):
        await service._call_cheap_llm("sys", "user", json_mode=True)
        assert registry.llm_service.run.await_args.kwargs["max_tokens"] == 150

    async def test_a_plain_answer_gets_a_small_budget(self, service, registry):
        await service._call_cheap_llm("sys", "user")
        assert registry.llm_service.run.await_args.kwargs["max_tokens"] == 50
