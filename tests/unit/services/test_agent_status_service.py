"""Tests for app.services.agent_status_service.

This is the one store that does not go through TenantCollection: agent run
status lives in Redis, keyed by a `status_key` that arrives verbatim from the
client's `x-status-key` header. That combination -- a client-chosen key and a
hand-rolled key format -- is why the isolation tests here carry as much weight
as the state-machine ones.
"""

from __future__ import annotations

import json

import pytest

from app.services.agent_status_service import (
    EMOJI_MAP,
    MESSAGES_MAP,
    AgentStatus,
    AgentStatusService,
)

ORG_ID = "org_test_0001"
OTHER_ORG = "org_other_9999"
KEY = "status-key-1"
AGENT_ID = "agent_1"
CHAT_ID = "chat_1"


@pytest.fixture
def service(fake_redis) -> AgentStatusService:
    return AgentStatusService(ORG_ID, fake_redis)


@pytest.fixture
def other_org_service(fake_redis) -> AgentStatusService:
    """A second tenant sharing the same Redis, as in production."""
    return AgentStatusService(OTHER_ORG, fake_redis)


# --- tenant isolation -----------------------------------------------------


class TestTenantIsolation:
    async def test_another_organization_cannot_read_the_status(
        self, service, other_org_service
    ):
        """The status key comes straight from a request header, so two orgs
        picking the same one must not collide -- one of them would be reading
        the other's run."""
        await service.start_session(KEY, AGENT_ID, chat_id=CHAT_ID)
        assert await other_org_service.get_status(KEY) is None

    async def test_each_organization_keeps_its_own_status_under_one_key(
        self, service, other_org_service
    ):
        await service.start_session(KEY, "mine", chat_id=CHAT_ID)
        await other_org_service.start_session(KEY, "theirs", chat_id=CHAT_ID)
        assert (await service.get_status(KEY))["agent_id"] == "mine"
        assert (await other_org_service.get_status(KEY))["agent_id"] == "theirs"

    async def test_the_redis_key_carries_the_organization(self, service):
        assert ORG_ID in service._get_redis_key(KEY)

    async def test_the_pubsub_channel_carries_the_organization(
        self, service, other_org_service
    ):
        """A shared channel would stream one org's live updates into the
        other's status WebSocket."""
        assert service._get_channel_name(KEY) != other_org_service._get_channel_name(
            KEY
        )

    async def test_finishing_one_organizations_session_leaves_the_other_running(
        self, service, other_org_service
    ):
        await service.start_session(KEY, "mine")
        await other_org_service.start_session(KEY, "theirs")
        await service.finish_session(KEY)
        assert (await other_org_service.get_status(KEY))["status"] == "in_progress"


# --- reads ----------------------------------------------------------------


class TestGetStatus:
    async def test_an_unknown_key_yields_nothing(self, service):
        assert await service.get_status("nope") is None

    async def test_a_started_session_is_readable(self, service):
        await service.start_session(KEY, AGENT_ID)
        assert (await service.get_status(KEY))["status_key"] == KEY

    async def test_unparseable_data_yields_nothing(self, service, fake_redis):
        """A half-written or hand-edited value must not take down the status
        endpoint for the whole run."""
        await fake_redis.set(service._get_redis_key(KEY), "not json")
        assert await service.get_status(KEY) is None


# --- session lifecycle ----------------------------------------------------


class TestStartSession:
    async def test_a_session_starts_in_progress(self, service):
        assert (await service.start_session(KEY, AGENT_ID)).status == "in_progress"

    async def test_the_agent_is_recorded_as_active(self, service):
        status = await service.start_session(KEY, AGENT_ID)
        assert status.agent_id == AGENT_ID and status.active_agent == AGENT_ID

    async def test_a_chat_id_is_optional(self, service):
        assert (await service.start_session(KEY, AGENT_ID)).chat_id is None

    async def test_a_chat_id_is_stored_when_given(self, service):
        status = await service.start_session(KEY, AGENT_ID, chat_id=CHAT_ID)
        assert status.chat_id == CHAT_ID

    async def test_a_new_session_starts_with_no_steps(self, service):
        assert (await service.start_session(KEY, AGENT_ID)).steps == []

    async def test_starting_again_replaces_the_previous_session(self, service):
        await service.start_session(KEY, "first")
        await service.update_step(KEY, "route_query")
        restarted = await service.start_session(KEY, "second")
        assert restarted.steps == [] and restarted.agent_id == "second"

    async def test_the_session_is_published(self, service, fake_redis):
        pubsub = fake_redis.pubsub()
        await pubsub.subscribe(service._get_channel_name(KEY))
        await service.start_session(KEY, AGENT_ID)
        # First frame is the subscribe confirmation, second is the payload.
        await pubsub.get_message(timeout=1)
        message = await pubsub.get_message(timeout=1)
        assert json.loads(message["data"])["status"] == "in_progress"


class TestUpdateStep:
    async def test_updating_an_unknown_session_yields_nothing(self, service):
        assert await service.update_step("nope", "route_query") is None

    async def test_a_step_is_appended(self, service):
        await service.start_session(KEY, AGENT_ID)
        status = await service.update_step(KEY, "route_query")
        assert [s.step for s in status.steps] == ["route_query"]

    async def test_the_current_step_tracks_the_latest(self, service):
        await service.start_session(KEY, AGENT_ID)
        await service.update_step(KEY, "route_query")
        status = await service.update_step(KEY, "generate_response")
        assert status.current_step == "generate_response"

    async def test_the_previous_step_is_closed_off(self, service):
        """An open step with no completion time renders as still-running in
        the UI, so a stale one would show two things happening at once."""
        await service.start_session(KEY, AGENT_ID)
        await service.update_step(KEY, "route_query")
        status = await service.update_step(KEY, "generate_response")
        assert status.steps[0].completed_at is not None
        assert status.steps[1].completed_at is None

    async def test_the_step_falls_back_to_the_session_agent(self, service):
        await service.start_session(KEY, AGENT_ID)
        status = await service.update_step(KEY, "route_query")
        assert status.steps[0].agent_id == AGENT_ID

    async def test_a_delegated_step_records_the_sub_agent(self, service):
        """Delegation hands the run to another agent; the status has to say
        which one is working or the UI names the wrong agent."""
        await service.start_session(KEY, AGENT_ID)
        status = await service.update_step(KEY, "delegated_task", "sub_agent")
        assert status.active_agent == "sub_agent"

    async def test_the_update_is_persisted(self, service):
        await service.start_session(KEY, AGENT_ID)
        await service.update_step(KEY, "route_query")
        assert (await service.get_status(KEY))["current_step"] == "route_query"

    async def test_the_update_is_published(self, service, fake_redis):
        await service.start_session(KEY, AGENT_ID)
        pubsub = fake_redis.pubsub()
        await pubsub.subscribe(service._get_channel_name(KEY))
        await service.update_step(KEY, "route_query")
        await pubsub.get_message(timeout=1)
        message = await pubsub.get_message(timeout=1)
        assert json.loads(message["data"])["current_step"] == "route_query"


class TestFinishSession:
    async def test_finishing_an_unknown_session_yields_nothing(self, service):
        assert await service.finish_session("nope") is None

    async def test_a_successful_run_completes(self, service):
        await service.start_session(KEY, AGENT_ID)
        assert (await service.finish_session(KEY)).status == "completed"

    async def test_a_failed_run_is_marked_failed(self, service):
        await service.start_session(KEY, AGENT_ID)
        assert (await service.finish_session(KEY, success=False)).status == "failed"

    async def test_open_steps_are_closed(self, service):
        await service.start_session(KEY, AGENT_ID)
        await service.update_step(KEY, "route_query")
        status = await service.finish_session(KEY)
        assert all(s.completed_at for s in status.steps)

    async def test_the_current_step_is_cleared(self, service):
        """Leaving one set would keep a spinner running after the answer has
        already arrived."""
        await service.start_session(KEY, AGENT_ID)
        await service.update_step(KEY, "route_query")
        status = await service.finish_session(KEY)
        assert status.current_step is None and status.current_message is None

    async def test_the_chat_id_can_be_attached_at_the_end(self, service):
        """A new chat has no id until the run creates one, so the final frame
        is the first chance to tell the client where the answer landed."""
        await service.start_session(KEY, AGENT_ID)
        status = await service.finish_session(KEY, chat_id=CHAT_ID)
        assert status.chat_id == CHAT_ID

    async def test_an_existing_chat_id_survives_being_omitted(self, service):
        await service.start_session(KEY, AGENT_ID, chat_id=CHAT_ID)
        assert (await service.finish_session(KEY)).chat_id == CHAT_ID

    async def test_the_final_state_is_kept_briefly(self, service, fake_redis):
        """Dropping it at once would race the client's last read."""
        await service.start_session(KEY, AGENT_ID)
        await service.finish_session(KEY)
        assert await fake_redis.ttl(service._get_redis_key(KEY)) > 0

    async def test_the_finish_is_published(self, service, fake_redis):
        await service.start_session(KEY, AGENT_ID)
        pubsub = fake_redis.pubsub()
        await pubsub.subscribe(service._get_channel_name(KEY))
        await service.finish_session(KEY)
        await pubsub.get_message(timeout=1)
        message = await pubsub.get_message(timeout=1)
        assert json.loads(message["data"])["status"] == "completed"


# --- step messages --------------------------------------------------------


class TestStepMessages:
    @pytest.mark.parametrize("step", sorted(MESSAGES_MAP))
    def test_every_known_step_renders_with_its_emoji(self, service, step):
        assert EMOJI_MAP[step] in service._generate_step_message(step, AGENT_ID)

    def test_an_unknown_step_still_renders(self, service):
        message = service._generate_step_message("some_new_step", AGENT_ID)
        assert "🤖" in message and "processing" in message

    def test_the_message_names_the_agent(self, service):
        assert AGENT_ID in service._generate_step_message("route_query", AGENT_ID)

    def test_the_message_is_drawn_from_the_step_templates(self, service):
        message = service._generate_step_message("route_query", AGENT_ID)
        assert any(t in message for t in MESSAGES_MAP["route_query"])


class TestAgentNameResolution:
    def test_a_missing_agent_id_reads_as_the_default_agent(self, service):
        assert service._get_agent_name("") == "KitaAgent"

    def test_the_default_agent_is_not_looked_up(self, service):
        assert service._get_agent_name("KitaAgent") == "KitaAgent"

    def test_the_id_is_used_when_there_is_no_agent_service(self, service):
        assert service._get_agent_name(AGENT_ID) == AGENT_ID

    def test_a_known_agent_resolves_to_its_name(self, service):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        # SimpleNamespace rather than MagicMock: `name` is the one attribute
        # MagicMock reserves for its own repr, so it cannot be set positionally.
        agent_service = MagicMock()
        agent_service.get_agent.return_value = SimpleNamespace(name="Scribe")
        service.set_agent_service(agent_service)
        assert service._get_agent_name(AGENT_ID) == "Scribe"

    def test_an_unknown_agent_falls_back_to_the_id(self, service):
        from unittest.mock import MagicMock

        agent_service = MagicMock()
        agent_service.get_agent.return_value = None
        service.set_agent_service(agent_service)
        assert service._get_agent_name(AGENT_ID) == AGENT_ID

    def test_a_lookup_failure_does_not_break_the_status(self, service):
        """A status update is telemetry; it must never be the reason a run
        fails."""
        from unittest.mock import MagicMock

        agent_service = MagicMock()
        agent_service.get_agent.side_effect = RuntimeError("mongo down")
        service.set_agent_service(agent_service)
        assert service._get_agent_name(AGENT_ID) == AGENT_ID


# --- model ----------------------------------------------------------------


class TestAgentStatusModel:
    async def test_a_persisted_status_round_trips(self, service):
        await service.start_session(KEY, AGENT_ID)
        await service.update_step(KEY, "route_query")
        stored = await service.get_status(KEY)
        assert AgentStatus.model_validate(stored).current_step == "route_query"
