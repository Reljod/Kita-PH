"""Tests for app.services.chat_service.

The interesting surface is the preview builder (it walks pydantic-ai message
parts, whose shape varies by part kind) and the id/ownership checks on
continue_chat — that is what stops one agent's chat being continued through
another agent's route.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.exceptions import (
    AgentRunStreamFailedError,
    ChatNotFoundError,
    KitaValidationError,
)
from app.models.chat import ChatContinueRequest, ChatCreateRequest, ChatResponse
from app.services.chat_service import ChatService, format_chat_response

OID = ObjectId("64b7f1c2e4b0a1a2b3c4d5e6")
NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)
AGENT = "agent_1"
ORG_ID = "org_test_0001"


def a_run_result(messages=None) -> MagicMock:
    result = MagicMock(name="run_result")
    result.all_messages.return_value = messages or []
    return result


def chat_doc(**overrides) -> dict:
    doc = {
        "_id": OID,
        "messages": [{"content": "hello there"}],
        "agent_id": AGENT,
        "created_at": NOW,
        "updated_at": NOW,
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def agent_service() -> MagicMock:
    service = MagicMock(name="agent_service")
    result = MagicMock()
    result.all_messages.return_value = [{"content": "hi"}, {"content": "hello"}]
    service.run = AsyncMock(return_value=result)

    async def run_stream(**kwargs):
        yield {"type": "content", "delta": "hi"}
        yield {"type": "result", "result": result}

    service.run_stream = run_stream
    return service


@pytest.fixture
def service(agent_service, mock_tenant_collection) -> ChatService:
    return ChatService(agent_service=agent_service, collection=mock_tenant_collection)


@pytest.fixture
def collection(service) -> MagicMock:
    return service.collection._collection


# --- Preview formatting ---------------------------------------------------


class TestFormatChatResponse:
    def test_maps_the_document(self):
        result = format_chat_response(chat_doc())
        assert isinstance(result, ChatResponse)
        assert result.id == str(OID) and result.agent_id == AGENT

    def test_preview_comes_from_the_first_message(self):
        doc = chat_doc(messages=[{"content": "first"}, {"content": "second"}])
        assert format_chat_response(doc).preview == "first"

    def test_a_long_preview_is_truncated_with_an_ellipsis(self):
        doc = chat_doc(messages=[{"content": "x" * 300}])
        preview = format_chat_response(doc).preview
        assert preview.endswith("...") and len(preview) == 103

    def test_a_short_preview_is_not_truncated(self):
        assert (
            format_chat_response(chat_doc(messages=[{"content": "short"}])).preview
            == "short"
        )

    def test_an_empty_chat_has_no_preview(self):
        assert format_chat_response(chat_doc(messages=[])).preview is None

    def test_an_empty_chat_keeps_an_empty_message_list(self):
        assert format_chat_response(chat_doc(messages=[])).messages == []

    def test_parts_are_joined_into_the_preview(self):
        doc = chat_doc(
            messages=[{"parts": [{"content": "line one"}, {"content": "line two"}]}]
        )
        assert format_chat_response(doc).preview == "line one\nline two"

    def test_thinking_parts_are_excluded_from_the_preview(self):
        """Chain-of-thought must not leak into a list view."""
        doc = chat_doc(
            messages=[
                {
                    "parts": [
                        {"content": "secret reasoning", "part_kind": "thinking"},
                        {"content": "the answer", "part_kind": "text"},
                    ]
                }
            ]
        )
        assert format_chat_response(doc).preview == "the answer"

    def test_a_message_with_only_thinking_parts_previews_as_empty(self):
        doc = chat_doc(
            messages=[{"parts": [{"content": "hidden", "part_kind": "thinking"}]}]
        )
        assert format_chat_response(doc).preview == ""

    def test_preview_only_drops_every_message_after_the_first(self):
        doc = chat_doc(messages=[{"content": "first"}, {"content": "second"}])
        assert format_chat_response(doc, preview_only=True).messages == [
            {"content": "first"}
        ]

    def test_the_full_view_keeps_every_message(self):
        doc = chat_doc(messages=[{"content": "first"}, {"content": "second"}])
        assert len(format_chat_response(doc).messages) == 2

    def test_a_non_dict_first_message_does_not_crash(self):
        assert format_chat_response(chat_doc(messages=["a bare string"])).preview == ""

    def test_a_message_without_content_previews_as_empty(self):
        assert format_chat_response(chat_doc(messages=[{"role": "user"}])).preview == ""

    def test_a_part_without_content_does_not_crash(self):
        """pydantic-ai tool-call parts carry `args`, not `content`; indexing
        blindly would turn a normal chat into a 500 in the list view."""
        doc = chat_doc(
            messages=[{"parts": [{"part_kind": "tool-call", "args": {"q": "x"}}]}]
        )
        assert format_chat_response(doc).preview is not None


# --- create_chat ----------------------------------------------------------


class TestCreateChat:
    async def test_requires_an_agent_id(self, service):
        with pytest.raises(KitaValidationError):
            await service.create_chat(ChatCreateRequest(message="hi"))

    async def test_runs_the_agent_with_the_message(self, service, agent_service):
        await service.create_chat(ChatCreateRequest(message="hi"), agent_id=AGENT)
        assert agent_service.run.await_args.kwargs["query"] == "hi"

    async def test_starts_with_no_message_history(self, service, agent_service):
        await service.create_chat(ChatCreateRequest(message="hi"), agent_id=AGENT)
        assert agent_service.run.await_args.kwargs["message_history"] is None

    async def test_persists_the_chat(self, service, collection):
        await service.create_chat(ChatCreateRequest(message="hi"), agent_id=AGENT)
        collection.insert_one.assert_called_once()

    async def test_the_write_is_org_scoped(self, service, collection):
        await service.create_chat(ChatCreateRequest(message="hi"), agent_id=AGENT)
        assert (
            collection.insert_one.call_args[0][0]["org_id"] == service.collection.org_id
        )

    async def test_a_versioned_agent_id_is_stored_as_its_base(
        self, service, collection
    ):
        """Chats belong to the agent, not to one pinned version of it."""
        await service.create_chat(
            ChatCreateRequest(message="hi"), agent_id="agent_1-v3"
        )
        assert collection.insert_one.call_args[0][0]["agent_id"] == "agent_1"

    async def test_the_status_key_is_forwarded(self, service, agent_service):
        await service.create_chat(
            ChatCreateRequest(message="hi"), agent_id=AGENT, status_key="k1"
        )
        assert agent_service.run.await_args.kwargs["status_key"] == "k1"

    async def test_returns_the_created_chat(self, service):
        result = await service.create_chat(
            ChatCreateRequest(message="hi"), agent_id=AGENT
        )
        assert isinstance(result, ChatResponse) and result.agent_id == AGENT


# --- continue_chat --------------------------------------------------------


class TestContinueChat:
    @pytest.mark.parametrize("bad_id", ["", "nope", "12345"])
    async def test_a_malformed_chat_id_is_not_found(self, service, bad_id):
        with pytest.raises(ChatNotFoundError):
            await service.continue_chat(bad_id, ChatContinueRequest(message="more"))

    async def test_an_unknown_chat_is_not_found(self, service, collection):
        collection.find_one.return_value = None
        with pytest.raises(ChatNotFoundError):
            await service.continue_chat(str(OID), ChatContinueRequest(message="more"))

    async def test_the_lookup_is_org_scoped(self, service, collection):
        collection.find_one.return_value = None
        with pytest.raises(ChatNotFoundError):
            await service.continue_chat(str(OID), ChatContinueRequest(message="more"))
        assert (
            collection.find_one.call_args[0][0]["org_id"] == service.collection.org_id
        )

    async def test_an_agent_id_scopes_the_lookup_to_that_agents_chats(
        self, service, collection
    ):
        """Passing an agent_id must narrow the query, so one agent's route
        cannot continue a chat that belongs to another."""
        collection.find_one.return_value = None
        with pytest.raises(ChatNotFoundError):
            await service.continue_chat(
                str(OID), ChatContinueRequest(message="more"), agent_id=AGENT
            )
        assert collection.find_one.call_args[0][0]["agent_id"] == AGENT

    async def test_a_chat_with_no_agent_and_no_override_is_rejected(
        self, service, collection
    ):
        collection.find_one.return_value = chat_doc(agent_id=None, messages=[])
        with pytest.raises(KitaValidationError):
            await service.continue_chat(str(OID), ChatContinueRequest(message="more"))

    async def test_the_chats_own_agent_is_used_when_none_is_given(
        self, service, collection, agent_service, monkeypatch
    ):
        monkeypatch.setattr(
            "app.services.chat_service.ModelMessagesTypeAdapter.validate_python",
            lambda messages: messages,
        )
        collection.find_one.return_value = chat_doc(messages=[])
        await service.continue_chat(str(OID), ChatContinueRequest(message="more"))
        assert agent_service.run.await_args.kwargs["agent_id"] == AGENT

    async def test_the_prior_history_is_replayed_into_the_agent(
        self, service, collection, agent_service, monkeypatch
    ):
        history = [{"content": "earlier"}]
        monkeypatch.setattr(
            "app.services.chat_service.ModelMessagesTypeAdapter.validate_python",
            lambda messages: messages,
        )
        collection.find_one.return_value = chat_doc(messages=history)
        await service.continue_chat(str(OID), ChatContinueRequest(message="more"))
        assert agent_service.run.await_args.kwargs["message_history"] == history

    async def test_the_chat_is_updated_with_the_new_transcript(
        self, service, collection, monkeypatch
    ):
        monkeypatch.setattr(
            "app.services.chat_service.ModelMessagesTypeAdapter.validate_python",
            lambda messages: messages,
        )
        collection.find_one.return_value = chat_doc(messages=[])
        await service.continue_chat(str(OID), ChatContinueRequest(message="more"))
        update = collection.update_one.call_args[0][1]["$set"]
        assert "messages" in update and isinstance(update["updated_at"], datetime)


# --- Reads ----------------------------------------------------------------


class TestGetChat:
    def test_returns_the_chat(self, service, collection):
        collection.find_one.return_value = chat_doc()
        assert service.get_chat(str(OID)).id == str(OID)

    def test_an_absent_chat_raises_not_found(self, service, collection):
        collection.find_one.return_value = None
        with pytest.raises(ChatNotFoundError):
            service.get_chat(str(OID))

    def test_a_malformed_id_raises_not_found_rather_than_leaking_bson(self, service):
        """bson.InvalidId escaping here would be a 500; the caller should see
        the same 404 as any other missing chat."""
        with pytest.raises(ChatNotFoundError):
            service.get_chat("not-an-id")

    def test_the_not_found_error_is_a_404(self, service):
        with pytest.raises(ChatNotFoundError) as excinfo:
            service.get_chat("not-an-id")
        assert excinfo.value.status_code == 404

    def test_an_agent_id_narrows_the_lookup(self, service, collection):
        collection.find_one.return_value = chat_doc()
        service.get_chat(str(OID), agent_id=AGENT)
        assert collection.find_one.call_args[0][0]["agent_id"] == AGENT


class TestGetAllChats:
    def test_returns_every_chat(self, service, collection, make_cursor):
        collection.find.return_value = make_cursor([chat_doc(), chat_doc()])
        assert len(service.get_all_chats()) == 2

    def test_returns_an_empty_list_when_there_are_none(
        self, service, collection, make_cursor
    ):
        collection.find.return_value = make_cursor([])
        assert service.get_all_chats() == []

    def test_the_query_is_org_scoped(self, service, collection, make_cursor):
        collection.find.return_value = make_cursor([])
        service.get_all_chats()
        assert collection.find.call_args[0][0]["org_id"] == service.collection.org_id

    def test_an_agent_filter_is_applied(self, service, collection, make_cursor):
        collection.find.return_value = make_cursor([])
        service.get_all_chats(agent_id=AGENT)
        assert collection.find.call_args[0][0]["agent_id"] == AGENT

    def test_preview_mode_returns_only_the_first_message(
        self, service, collection, make_cursor
    ):
        collection.find.return_value = make_cursor(
            [chat_doc(messages=[{"content": "a"}, {"content": "b"}])]
        )
        assert len(service.get_all_chats(preview=True)[0].messages) == 1


# --- agent version resolution ---------------------------------------------


class TestPinnedAgentVersions:
    """Chats are written against the *base* agent id, and get_all_chats
    already normalises before querying. The single-chat lookups did not, so a
    caller pinned to "<base>-v2" -- a form the routes accept and the rest of
    the codebase resolves -- was told its own chats did not exist.
    """

    @pytest.fixture
    def real_service(self, agent_service, mongo_db):
        from app.db import TenantCollection

        return ChatService(agent_service, TenantCollection(mongo_db["chats"], ORG_ID))

    @pytest.fixture
    def seeded(self, mongo_db):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return str(
            mongo_db["chats"]
            .insert_one(
                {
                    "org_id": ORG_ID,
                    "agent_id": AGENT,
                    "messages": [],
                    "created_at": now,
                    "updated_at": now,
                }
            )
            .inserted_id
        )

    def test_a_pinned_version_reads_its_own_chat(self, real_service, seeded):
        assert real_service.get_chat(seeded, agent_id=f"{AGENT}-v2")

    def test_a_pinned_version_lists_its_own_chats(self, real_service, seeded):
        assert len(real_service.get_all_chats(agent_id=f"{AGENT}-v2")) == 1

    async def test_a_pinned_version_continues_its_own_chat(self, real_service, seeded):
        assert await real_service.continue_chat(
            seeded, ChatContinueRequest(message="more"), agent_id=f"{AGENT}-v2"
        )

    async def test_a_pinned_version_continues_its_own_chat_streaming(
        self, real_service, seeded
    ):
        frames = [
            frame
            async for frame in real_service.continue_chat_stream(
                seeded, ChatContinueRequest(message="more"), agent_id=f"{AGENT}-v2"
            )
        ]
        assert frames[-1]["type"] == "done"

    def test_a_genuinely_different_agent_is_still_refused(self, real_service, seeded):
        """Normalising must not widen the check into no check at all."""
        with pytest.raises(ChatNotFoundError):
            real_service.get_chat(seeded, agent_id="6a67000000000000000000ff")


# --- streaming ------------------------------------------------------------


def a_streaming_agent(*frames):
    """An agent service whose run_stream replays the given frames."""
    service = MagicMock(name="agent_service")

    async def run_stream(**kwargs):
        for frame in frames:
            yield frame

    service.run_stream = run_stream
    return service


class TestChatStreams:
    @pytest.fixture
    def streamed(self, mongo_db):
        from app.db import TenantCollection

        def _make(*frames):
            return ChatService(
                a_streaming_agent(*frames),
                TenantCollection(mongo_db["chats"], ORG_ID),
            )

        return _make

    async def test_content_frames_reach_the_caller(self, streamed):
        service = streamed(
            {"type": "content", "delta": "hi"},
            {"type": "result", "result": a_run_result()},
        )
        frames = [
            f
            async for f in service.create_chat_stream(
                ChatCreateRequest(message="hi"), agent_id=AGENT
            )
        ]
        assert {"type": "content", "delta": "hi"} in frames

    async def test_the_raw_result_frame_is_not_forwarded(self, streamed):
        """It carries the run object, which does not serialise to SSE."""
        service = streamed(
            {"type": "content", "delta": "hi"},
            {"type": "result", "result": a_run_result()},
        )
        frames = [
            f
            async for f in service.create_chat_stream(
                ChatCreateRequest(message="hi"), agent_id=AGENT
            )
        ]
        assert all(f["type"] != "result" for f in frames)

    async def test_the_stream_ends_with_the_saved_chat(self, streamed, mongo_db):
        """The client needs the id to navigate to what it just started."""
        service = streamed({"type": "result", "result": a_run_result()})
        frames = [
            f
            async for f in service.create_chat_stream(
                ChatCreateRequest(message="hi"), agent_id=AGENT
            )
        ]
        assert frames[-1]["type"] == "done"
        assert frames[-1]["chat"]["id"] == str(mongo_db["chats"].find_one({})["_id"])

    async def test_a_stream_is_required_to_produce_a_result(self, streamed, mongo_db):
        """No result means no transcript to save; persisting an empty chat
        would strand the user's message with no answer."""
        service = streamed({"type": "content", "delta": "partial"})
        with pytest.raises(AgentRunStreamFailedError):
            [
                f
                async for f in service.create_chat_stream(
                    ChatCreateRequest(message="hi"), agent_id=AGENT
                )
            ]
        assert mongo_db["chats"].count_documents({}) == 0

    async def test_streaming_requires_an_agent(self, streamed):
        service = streamed({"type": "result", "result": a_run_result()})
        with pytest.raises(KitaValidationError):
            [
                f
                async for f in service.create_chat_stream(
                    ChatCreateRequest(message="x")
                )
            ]

    async def test_a_versioned_agent_id_is_stored_as_its_base(self, streamed, mongo_db):
        service = streamed({"type": "result", "result": a_run_result()})
        [
            f
            async for f in service.create_chat_stream(
                ChatCreateRequest(message="hi"), agent_id=f"{AGENT}-v3"
            )
        ]
        assert mongo_db["chats"].find_one({})["agent_id"] == AGENT
