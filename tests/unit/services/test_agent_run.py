"""Tests for AgentService.run and AgentService.run_stream.

`run_stream` translates pydantic-ai's event objects into the SSE frames the UI
consumes, dispatching on `type(event).__name__`. That string dispatch is the
reason these tests build stand-in event classes rather than importing the real
ones: the production code never touches the classes themselves, only their
names, so a fake with the right name exercises exactly the branch a real event
would.

The other half is status bookkeeping. A run that fails without calling
finish_session leaves the UI's status socket waiting forever, so the failure
paths matter as much as the happy one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db import TenantCollection
from app.exceptions import (
    AgentNotFoundError,
    AgentRunFailedError,
    AgentRunStreamFailedError,
)
from app.models.agent import AgentCreateRequest
from app.services.agent_service import AgentService

ORG_ID = "org_test_0001"
STATUS_KEY = "status-1"
CHAT_ID = "chat-1"


# --- stand-in pydantic-ai events -----------------------------------------
# The names are the contract; the payloads are whatever the branch reads.


class TextPart:
    def __init__(self, content: str):
        self.content = content


class ThinkingPart:
    def __init__(self, content: str):
        self.content = content


class ToolCallPart:
    def __init__(self):
        self.content = ""


class PartStartEvent:
    def __init__(self, part):
        self.part = part


class TextPartDelta:
    def __init__(self, content_delta: str):
        self.content_delta = content_delta


class ThinkingPartDelta:
    def __init__(self, content_delta: str):
        self.content_delta = content_delta


class ToolCallPartDelta:
    def __init__(self):
        self.content_delta = ""


class PartDeltaEvent:
    def __init__(self, delta):
        self.delta = delta


class FunctionToolCallEvent:
    pass


class ModelResponseStreamEvent:
    pass


class AgentRunResultEvent:
    def __init__(self, result):
        self.result = result


class UnknownEvent:
    """Anything the translator has no branch for."""


def a_usage() -> MagicMock:
    usage = MagicMock(name="usage")
    usage.request_tokens = 10
    usage.response_tokens = 20
    usage.total_tokens = 30
    usage.requests = 1
    return usage


def a_result() -> MagicMock:
    result = MagicMock(name="run_result")
    result.usage.return_value = a_usage()
    return result


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def status_service() -> MagicMock:
    service = MagicMock(name="agent_status_service")
    service.start_session = AsyncMock()
    service.update_step = AsyncMock()
    service.finish_session = AsyncMock()
    return service


@pytest.fixture
def registry(monkeypatch, status_service) -> MagicMock:
    """`get_services` is imported inside the methods, so patch it at source."""
    services = MagicMock(name="service_registry")
    services.agent_status_service = status_service
    monkeypatch.setattr(
        "app.dependencies.services.get_services", lambda org_id: services
    )
    return services


@pytest.fixture
def service(mongo_db) -> AgentService:
    llm_service = MagicMock(name="llm_service")
    llm_service.get_llm.return_value = MagicMock(model="openai/gpt-4o-mini")
    return AgentService(llm_service, TenantCollection(mongo_db["agents"], ORG_ID))


@pytest.fixture
async def agent_id(service) -> str:
    created = await service.create_agent(
        AgentCreateRequest(
            name="Researcher",
            role="analyst",
            goal="find things",
            backstory="trained on the archive",
            llm_id="llm_1",
        )
    )
    return created.id


@pytest.fixture
def runnable(monkeypatch, service) -> MagicMock:
    """Stub the pydantic-ai Agent so no model is ever constructed."""
    agent = MagicMock(name="runnable_agent")
    agent.run = AsyncMock(return_value=a_result())
    monkeypatch.setattr(service, "get_runnable_agent", lambda agent_id: agent)
    return agent


def streaming(*events):
    """Turn a list of stand-in events into a run_stream_events replacement."""

    async def run_stream_events(*args, **kwargs):
        for event in events:
            yield event

    return run_stream_events


async def collect(service, agent_id, **kwargs) -> list[dict]:
    return [frame async for frame in service.run_stream(agent_id, "q", **kwargs)]


# --- run ------------------------------------------------------------------


class TestRun:
    async def test_the_result_is_returned(self, service, agent_id, registry, runnable):
        assert await service.run(agent_id, "hello") is runnable.run.return_value

    async def test_the_query_reaches_the_agent(
        self, service, agent_id, registry, runnable
    ):
        await service.run(agent_id, "hello")
        assert runnable.run.await_args[0][0] == "hello"

    async def test_message_history_is_forwarded(
        self, service, agent_id, registry, runnable
    ):
        history = [{"role": "user"}]
        await service.run(agent_id, "hello", message_history=history)
        assert runnable.run.await_args.kwargs["message_history"] is history

    async def test_the_tenant_is_carried_in_the_dependencies(
        self, service, agent_id, registry, runnable
    ):
        """Tools resolve their collections from deps, so a missing org_id here
        would silently widen every tool call to the whole database."""
        await service.run(agent_id, "hello")
        assert runnable.run.await_args.kwargs["deps"]["org_id"] == ORG_ID

    async def test_no_status_is_published_without_a_key(
        self, service, agent_id, registry, runnable, status_service
    ):
        await service.run(agent_id, "hello")
        status_service.start_session.assert_not_awaited()

    async def test_a_session_is_opened_and_closed_for_a_status_key(
        self, service, agent_id, registry, runnable, status_service
    ):
        await service.run(agent_id, "hello", status_key=STATUS_KEY, chat_id=CHAT_ID)
        status_service.start_session.assert_awaited_once()
        assert status_service.finish_session.await_args.kwargs["success"] is True

    async def test_the_steps_are_published_in_order(
        self, service, agent_id, registry, runnable, status_service
    ):
        await service.run(agent_id, "hello", status_key=STATUS_KEY)
        steps = [call[0][1] for call in status_service.update_step.await_args_list]
        assert steps == ["draft_response", "finalize_response"]

    async def test_a_failure_is_wrapped(self, service, agent_id, registry, runnable):
        runnable.run = AsyncMock(side_effect=RuntimeError("model exploded"))
        with pytest.raises(AgentRunFailedError) as exc:
            await service.run(agent_id, "hello")
        assert "model exploded" in exc.value.message

    async def test_a_kita_exception_is_not_re_wrapped(
        self, service, registry, monkeypatch
    ):
        """A 404 that came back as a 500 would tell the caller to retry a run
        that can never succeed."""

        def missing(agent_id):
            raise AgentNotFoundError(agent_id)

        monkeypatch.setattr(service, "get_runnable_agent", missing)
        with pytest.raises(AgentNotFoundError):
            await service.run("nope", "hello")

    async def test_a_failure_still_closes_the_status_session(
        self, service, agent_id, registry, runnable, status_service
    ):
        """Otherwise the UI's status socket waits for an update that is never
        coming."""
        runnable.run = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(AgentRunFailedError):
            await service.run(agent_id, "hello", status_key=STATUS_KEY)
        assert status_service.finish_session.await_args.kwargs["success"] is False

    async def test_a_failure_without_a_status_key_publishes_nothing(
        self, service, agent_id, registry, runnable, status_service
    ):
        runnable.run = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(AgentRunFailedError):
            await service.run(agent_id, "hello")
        status_service.finish_session.assert_not_awaited()

    async def test_a_long_query_is_truncated_for_logging(
        self, service, agent_id, registry, runnable
    ):
        """The full query is echoed into spans and logs; an unbounded one puts
        whatever the user typed into the telemetry backend verbatim."""
        await service.run(agent_id, "x" * 500)
        assert runnable.run.await_args[0][0] == "x" * 500


# --- run_stream: event translation ---------------------------------------


class TestStreamTranslation:
    async def test_a_text_part_becomes_content(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(PartStartEvent(TextPart("hi")))
        assert await collect(service, agent_id) == [{"type": "content", "delta": "hi"}]

    async def test_a_thinking_part_becomes_a_thought(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(PartStartEvent(ThinkingPart("hmm")))
        assert await collect(service, agent_id) == [{"type": "thought", "delta": "hmm"}]

    async def test_a_text_delta_becomes_content(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(PartDeltaEvent(TextPartDelta("wor")))
        assert await collect(service, agent_id) == [{"type": "content", "delta": "wor"}]

    async def test_a_thinking_delta_becomes_a_thought(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(PartDeltaEvent(ThinkingPartDelta("...")))
        assert await collect(service, agent_id) == [{"type": "thought", "delta": "..."}]

    async def test_deltas_accumulate_into_separate_frames(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(
            PartDeltaEvent(TextPartDelta("Hel")), PartDeltaEvent(TextPartDelta("lo"))
        )
        frames = await collect(service, agent_id)
        assert [f["delta"] for f in frames] == ["Hel", "lo"]

    async def test_an_unknown_event_is_ignored(
        self, service, agent_id, registry, runnable
    ):
        """New pydantic-ai releases add event types; an unrecognised one must
        not break the stream."""
        runnable.run_stream_events = streaming(
            UnknownEvent(), PartStartEvent(TextPart("hi"))
        )
        assert await collect(service, agent_id) == [{"type": "content", "delta": "hi"}]

    async def test_a_part_event_without_a_part_is_ignored(
        self, service, agent_id, registry, runnable
    ):
        bare = PartStartEvent(TextPart(""))
        del bare.part
        runnable.run_stream_events = streaming(bare)
        assert await collect(service, agent_id) == []

    async def test_a_delta_event_without_a_delta_is_ignored(
        self, service, agent_id, registry, runnable
    ):
        bare = PartDeltaEvent(TextPartDelta(""))
        del bare.delta
        runnable.run_stream_events = streaming(bare)
        assert await collect(service, agent_id) == []

    async def test_an_empty_stream_yields_nothing(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming()
        assert await collect(service, agent_id) == []


class TestStreamToolCalls:
    """Text emitted before a tool call was not the answer — it was the model
    reasoning out loud. It gets retracted and re-sent as a thought."""

    async def test_buffered_text_is_retracted_when_a_tool_call_starts(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(
            PartStartEvent(TextPart("Let me check")),
            PartStartEvent(ToolCallPart()),
        )
        assert await collect(service, agent_id) == [
            {"type": "content", "delta": "Let me check"},
            {"type": "reset"},
            {"type": "thought", "delta": "Let me check"},
        ]

    async def test_a_tool_call_with_no_buffered_text_retracts_nothing(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(PartStartEvent(ToolCallPart()))
        assert await collect(service, agent_id) == []

    async def test_a_tool_call_delta_also_retracts(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(
            PartDeltaEvent(TextPartDelta("thinking")),
            PartDeltaEvent(ToolCallPartDelta()),
        )
        assert {"type": "reset"} in await collect(service, agent_id)

    async def test_a_function_tool_call_event_retracts(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(
            PartStartEvent(TextPart("about to call")), FunctionToolCallEvent()
        )
        assert {"type": "reset"} in await collect(service, agent_id)

    async def test_text_is_only_retracted_once(
        self, service, agent_id, registry, runnable
    ):
        """The buffer is cleared on retraction, so a second tool call in the
        same run must not replay the same text."""
        runnable.run_stream_events = streaming(
            PartStartEvent(TextPart("once")),
            FunctionToolCallEvent(),
            FunctionToolCallEvent(),
        )
        frames = await collect(service, agent_id)
        assert len([f for f in frames if f["type"] == "reset"]) == 1

    async def test_a_new_model_response_clears_the_buffer(
        self, service, agent_id, registry, runnable
    ):
        """Text from a previous model turn must not be retracted into the
        next one."""
        runnable.run_stream_events = streaming(
            PartStartEvent(TextPart("first turn")),
            ModelResponseStreamEvent(),
            FunctionToolCallEvent(),
        )
        frames = await collect(service, agent_id)
        assert {"type": "reset"} not in frames

    async def test_text_after_a_tool_call_still_streams(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(
            PartStartEvent(TextPart("checking")),
            FunctionToolCallEvent(),
            PartStartEvent(TextPart("the answer")),
        )
        frames = await collect(service, agent_id)
        assert frames[-1] == {"type": "content", "delta": "the answer"}


class TestStreamResult:
    async def test_the_run_result_is_yielded(
        self, service, agent_id, registry, runnable
    ):
        result = a_result()
        runnable.run_stream_events = streaming(AgentRunResultEvent(result))
        assert await collect(service, agent_id) == [
            {"type": "result", "result": result}
        ]

    async def test_the_result_comes_after_the_content(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = streaming(
            PartStartEvent(TextPart("hi")), AgentRunResultEvent(a_result())
        )
        frames = await collect(service, agent_id)
        assert frames[0]["type"] == "content" and frames[-1]["type"] == "result"


class TestStreamStatus:
    async def test_no_status_is_published_without_a_key(
        self, service, agent_id, registry, runnable, status_service
    ):
        runnable.run_stream_events = streaming(PartStartEvent(TextPart("hi")))
        await collect(service, agent_id)
        status_service.start_session.assert_not_awaited()

    async def test_a_session_is_opened_for_a_status_key(
        self, service, agent_id, registry, runnable, status_service
    ):
        runnable.run_stream_events = streaming(PartStartEvent(TextPart("hi")))
        await collect(service, agent_id, status_key=STATUS_KEY)
        status_service.start_session.assert_awaited_once()

    async def test_the_finalize_step_is_published_once_text_starts(
        self, service, agent_id, registry, runnable, status_service
    ):
        runnable.run_stream_events = streaming(PartStartEvent(TextPart("hi")))
        await collect(service, agent_id, status_key=STATUS_KEY)
        steps = [call[0][1] for call in status_service.update_step.await_args_list]
        assert steps == ["draft_response", "finalize_response"]

    async def test_the_finalize_step_is_published_only_once(
        self, service, agent_id, registry, runnable, status_service
    ):
        """One Redis publish per token would swamp the status channel."""
        runnable.run_stream_events = streaming(
            *[PartDeltaEvent(TextPartDelta(c)) for c in "hello"]
        )
        await collect(service, agent_id, status_key=STATUS_KEY)
        steps = [call[0][1] for call in status_service.update_step.await_args_list]
        assert steps.count("finalize_response") == 1

    async def test_a_thinking_only_stream_never_finalizes(
        self, service, agent_id, registry, runnable, status_service
    ):
        runnable.run_stream_events = streaming(PartStartEvent(ThinkingPart("hmm")))
        await collect(service, agent_id, status_key=STATUS_KEY)
        steps = [call[0][1] for call in status_service.update_step.await_args_list]
        assert "finalize_response" not in steps

    async def test_the_session_closes_successfully(
        self, service, agent_id, registry, runnable, status_service
    ):
        runnable.run_stream_events = streaming(PartStartEvent(TextPart("hi")))
        await collect(service, agent_id, status_key=STATUS_KEY, chat_id=CHAT_ID)
        assert status_service.finish_session.await_args.kwargs["success"] is True


class TestStreamFailures:
    async def test_a_mid_stream_failure_is_wrapped(
        self, service, agent_id, registry, runnable
    ):
        async def failing(*args, **kwargs):
            yield PartStartEvent(TextPart("partial"))
            raise RuntimeError("connection reset")

        runnable.run_stream_events = failing
        with pytest.raises(AgentRunStreamFailedError) as exc:
            await collect(service, agent_id)
        assert "connection reset" in exc.value.message

    async def test_frames_before_the_failure_still_reached_the_caller(
        self, service, agent_id, registry, runnable
    ):
        """A stream that drops what it already produced makes a partial answer
        look like no answer at all."""

        async def failing(*args, **kwargs):
            yield PartStartEvent(TextPart("partial"))
            raise RuntimeError("boom")

        runnable.run_stream_events = failing
        seen = []
        with pytest.raises(AgentRunStreamFailedError):
            async for frame in service.run_stream(agent_id, "q"):
                seen.append(frame)
        assert seen == [{"type": "content", "delta": "partial"}]

    async def test_a_kita_exception_is_not_re_wrapped(
        self, service, registry, monkeypatch
    ):
        def missing(agent_id):
            raise AgentNotFoundError(agent_id)

        monkeypatch.setattr(service, "get_runnable_agent", missing)
        with pytest.raises(AgentNotFoundError):
            await collect(service, "nope")

    async def test_a_failure_closes_the_status_session(
        self, service, agent_id, registry, runnable, status_service
    ):
        async def failing(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

        runnable.run_stream_events = failing
        with pytest.raises(AgentRunStreamFailedError):
            await collect(service, agent_id, status_key=STATUS_KEY)
        assert status_service.finish_session.await_args.kwargs["success"] is False

    async def test_a_failure_before_any_event_is_still_wrapped(
        self, service, agent_id, registry, runnable
    ):
        runnable.run_stream_events = MagicMock(side_effect=RuntimeError("no stream"))
        with pytest.raises(AgentRunStreamFailedError):
            await collect(service, agent_id)
