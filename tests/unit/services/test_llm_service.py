"""Tests for app.services.llm_service — LLM registry CRUD and the run() call."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId

from app.models.llm import LlmCreateRequest, LlmResponse
from app.services.llm_service import LlmService, format_llm_response

OID = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def llm_doc(**overrides) -> dict:
    doc = {
        "_id": OID,
        "name": "default",
        "model": "x-ai/grok-4.3",
        "provider": "openrouter",
        "created_at": NOW,
        "updated_at": NOW,
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def collection(mock_tenant_collection):
    return mock_tenant_collection


@pytest.fixture
def service(collection) -> LlmService:
    return LlmService(collection)


def completion(content: str | None, usage=None) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    result = MagicMock()
    result.choices = [choice]
    result.usage = usage
    return result


# --- Formatting -----------------------------------------------------------


class TestFormatLlmResponse:
    def test_maps_every_field(self):
        result = format_llm_response(llm_doc())
        assert result == LlmResponse(
            id=str(OID),
            name="default",
            model="x-ai/grok-4.3",
            provider="openrouter",
            created_at=NOW,
            updated_at=NOW,
        )

    def test_stringifies_the_object_id(self):
        assert isinstance(format_llm_response(llm_doc()).id, str)

    def test_raises_when_a_required_field_is_missing(self):
        doc = llm_doc()
        del doc["model"]
        with pytest.raises(KeyError):
            format_llm_response(doc)


# --- CRUD -----------------------------------------------------------------


class TestAddLlm:
    def test_returns_the_created_llm(self, service):
        result = service.add_llm(
            LlmCreateRequest(name="fast", model="x-ai/grok-4.3", provider="openrouter")
        )
        assert result.name == "fast" and result.id == str(OID)

    def test_persists_the_document(self, service, collection):
        service.add_llm(LlmCreateRequest(name="fast", model="m", provider="openrouter"))
        stored = collection._collection.insert_one.call_args[0][0]
        assert stored["name"] == "fast" and stored["model"] == "m"

    def test_the_write_is_org_scoped(self, service, collection):
        service.add_llm(LlmCreateRequest(name="fast", model="m"))
        stored = collection._collection.insert_one.call_args[0][0]
        assert stored["org_id"] == collection.org_id

    def test_provider_defaults_to_openrouter(self, service):
        assert (
            service.add_llm(LlmCreateRequest(name="f", model="m")).provider
            == "openrouter"
        )

    def test_timestamps_are_stamped(self, service, collection):
        service.add_llm(LlmCreateRequest(name="f", model="m"))
        stored = collection._collection.insert_one.call_args[0][0]
        assert isinstance(stored["created_at"], datetime)


class TestListLlms:
    def test_returns_every_llm(self, service, collection, make_cursor):
        collection._collection.find.return_value = make_cursor(
            [llm_doc(), llm_doc(name="b")]
        )
        assert len(service.list_llms()) == 2

    def test_returns_an_empty_list_when_none_exist(
        self, service, collection, make_cursor
    ):
        collection._collection.find.return_value = make_cursor([])
        assert service.list_llms() == []

    def test_sorts_newest_first(self, service, collection, make_cursor):
        cursor = make_cursor([llm_doc()])
        collection._collection.find.return_value = cursor
        service.list_llms()
        cursor.sort.assert_called_once_with("created_at", -1)

    def test_the_query_is_org_scoped(self, service, collection, make_cursor):
        collection._collection.find.return_value = make_cursor([])
        service.list_llms()
        assert collection._collection.find.call_args[0][0] == {
            "org_id": collection.org_id
        }


class TestGetLlm:
    def test_returns_the_llm(self, service, collection):
        collection._collection.find_one.return_value = llm_doc()
        assert service.get_llm(str(OID)).id == str(OID)

    def test_returns_none_when_absent(self, service, collection):
        collection._collection.find_one.return_value = None
        assert service.get_llm(str(OID)) is None

    def test_the_lookup_is_org_scoped(self, service, collection):
        collection._collection.find_one.return_value = llm_doc()
        service.get_llm(str(OID))
        assert (
            collection._collection.find_one.call_args[0][0]["org_id"]
            == collection.org_id
        )

    @pytest.mark.parametrize("bad_id", ["", "nope", "12345", "z" * 24])
    def test_rejects_a_malformed_id(self, service, bad_id):
        with pytest.raises(ValueError, match="Invalid LLM ID"):
            service.get_llm(bad_id)


class TestDeleteLlm:
    def test_returns_true_when_a_document_was_removed(self, service, collection):
        collection._collection.delete_one.return_value = MagicMock(deleted_count=1)
        assert service.delete_llm(str(OID)) is True

    def test_returns_false_when_nothing_matched(self, service, collection):
        collection._collection.delete_one.return_value = MagicMock(deleted_count=0)
        assert service.delete_llm(str(OID)) is False

    def test_the_delete_is_org_scoped(self, service, collection):
        service.delete_llm(str(OID))
        assert (
            collection._collection.delete_one.call_args[0][0]["org_id"]
            == collection.org_id
        )

    @pytest.mark.parametrize("bad_id", ["", "nope", "12345"])
    def test_rejects_a_malformed_id(self, service, bad_id):
        with pytest.raises(ValueError, match="Invalid LLM ID"):
            service.delete_llm(bad_id)

    def test_a_malformed_id_never_reaches_the_database(self, service, collection):
        with pytest.raises(ValueError):
            service.delete_llm("nope")
        collection._collection.delete_one.assert_not_called()


# --- run() ----------------------------------------------------------------


class TestRun:
    @pytest.fixture
    def client(self, service):
        create = AsyncMock(return_value=completion("  hello from the model  "))
        service.client = MagicMock()
        service.client.chat.completions.create = create
        return create

    async def test_returns_the_stripped_content(self, service, client):
        result = await service.run("m", [{"role": "user", "content": "hi"}])
        assert result == "hello from the model"

    async def test_passes_the_model_and_messages_through(self, service, client):
        messages = [{"role": "user", "content": "hi"}]
        await service.run("some/model", messages)
        assert client.call_args.kwargs["model"] == "some/model"
        assert client.call_args.kwargs["messages"] == messages

    async def test_json_mode_sets_the_response_format(self, service, client):
        await service.run("m", [{"role": "user", "content": "hi"}], json_mode=True)
        assert client.call_args.kwargs["response_format"] == {"type": "json_object"}

    async def test_response_format_is_none_by_default(self, service, client):
        await service.run("m", [{"role": "user", "content": "hi"}])
        assert client.call_args.kwargs["response_format"] is None

    async def test_temperature_and_max_tokens_are_forwarded(self, service, client):
        await service.run(
            "m", [{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=256
        )
        assert client.call_args.kwargs["temperature"] == 0.7
        assert client.call_args.kwargs["max_tokens"] == 256

    async def test_temperature_defaults_to_deterministic(self, service, client):
        await service.run("m", [{"role": "user", "content": "hi"}])
        assert client.call_args.kwargs["temperature"] == 0.0

    async def test_an_empty_message_list_is_still_sent(self, service, client):
        await service.run("m", [])
        assert client.call_args.kwargs["messages"] == []

    async def test_usage_metadata_is_tolerated_when_absent(self, service, client):
        client.return_value = completion("text", usage=None)
        assert await service.run("m", [{"role": "user", "content": "hi"}]) == "text"

    async def test_usage_metadata_is_read_when_present(self, service, client):
        usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        client.return_value = completion("text", usage=usage)
        assert await service.run("m", [{"role": "user", "content": "hi"}]) == "text"

    async def test_provider_errors_propagate(self, service, client):
        client.side_effect = RuntimeError("upstream 503")
        with pytest.raises(RuntimeError, match="upstream 503"):
            await service.run("m", [{"role": "user", "content": "hi"}])

    async def test_a_null_content_response_does_not_crash(self, service, client):
        """OpenRouter returns content=None when a model emits only tool calls
        or trips a content filter. Calling .strip() on that is an
        AttributeError that surfaces as a 500 rather than a handled result."""
        client.return_value = completion(None)
        result = await service.run("m", [{"role": "user", "content": "hi"}])
        assert result == ""

    async def test_a_long_prompt_is_truncated_for_logging_not_for_the_call(
        self, service, client
    ):
        long_prompt = "x" * 500
        await service.run("m", [{"role": "user", "content": long_prompt}])
        assert client.call_args.kwargs["messages"][0]["content"] == long_prompt


class TestRunStatusReporting:
    @pytest.fixture
    def client(self, service):
        service.client = MagicMock()
        service.client.chat.completions.create = AsyncMock(
            return_value=completion("ok")
        )
        return service.client.chat.completions.create

    async def test_status_is_updated_when_a_key_and_step_are_given(
        self, service, client
    ):
        status_service = MagicMock()
        status_service.update_step = AsyncMock()
        registry = MagicMock(agent_status_service=status_service)
        with patch("app.dependencies.services.get_services", return_value=registry):
            await service.run(
                "m", [], status_key="key-1", step="thinking", agent_id="A"
            )
        status_service.update_step.assert_awaited_once_with("key-1", "thinking", "A")

    async def test_status_is_skipped_without_a_step(self, service, client):
        with patch("app.dependencies.services.get_services") as get_services:
            await service.run("m", [], status_key="key-1")
        get_services.assert_not_called()

    async def test_status_is_skipped_without_a_key(self, service, client):
        with patch("app.dependencies.services.get_services") as get_services:
            await service.run("m", [], step="thinking")
        get_services.assert_not_called()

    async def test_a_status_failure_does_not_fail_the_llm_call(self, service, client):
        """Status reporting is best-effort telemetry; it must never take down
        the actual model call."""
        with patch(
            "app.dependencies.services.get_services",
            side_effect=RuntimeError("redis down"),
        ):
            assert await service.run("m", [], status_key="k", step="s") == "ok"
