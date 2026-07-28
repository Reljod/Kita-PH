"""Behaviour of the Telegram channel.

These run against an in-memory Mongo (`patched_db`) with a stubbed Telegram
client, so the assertions are about our own logic — routing, deduplication,
tenant isolation, who answers — rather than about httpx.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.exceptions import (
    IntegrationCredentialError,
    IntegrationNotConfiguredError,
    KitaValidationError,
    TelegramThreadNotFoundError,
)
from app.models.telegram import (
    TelegramConnectRequest,
    TelegramIntegrationUpdate,
    TelegramSender,
    TelegramThreadUpdate,
)
from app.services.webhook.telegram_client import TelegramApiError
from app.services.webhook.telegram_service import TelegramService, mask_token

BOT_TOKEN = "123456789:AAExampleTokenValueThatIsLongEnough"
OTHER_ORG = "org_test_0002"


@pytest.fixture
def telegram_client() -> AsyncMock:
    client = AsyncMock(name="telegram_client")
    client.get_me.return_value = {
        "id": 123456789,
        "username": "kita_test_bot",
        "first_name": "Kita Test",
    }
    client.set_webhook.return_value = True
    client.delete_webhook.return_value = True
    client.send_message.return_value = {"message_id": 4242}
    return client


@pytest.fixture
def service(telegram_client, patched_db) -> TelegramService:
    return TelegramService(client=telegram_client)


def update_payload(update_id: int, text: str, chat_id: int = 555) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id * 10,
            "text": text,
            "chat": {"id": chat_id, "type": "private", "first_name": "Ada"},
            "from": {"id": chat_id, "is_bot": False, "first_name": "Ada"},
        },
    }


def agent_result(text: str) -> MagicMock:
    result = MagicMock()
    result.output = text
    result.all_messages.return_value = []
    return result


@pytest.fixture
def agent_service() -> MagicMock:
    service = MagicMock(name="agent_service")
    service.run = AsyncMock(return_value=agent_result("Hello from the agent"))
    agent = MagicMock()
    agent.id = "agent_test_0001"
    service.get_all_agents.return_value = [agent]
    return service


@pytest.fixture
def patched_registry(agent_service):
    """Route `get_services(org).agent_service` to a stub.

    The service resolves the registry lazily inside the reply path, so the
    patch has to target the import site rather than the module attribute.
    """
    registry = MagicMock()
    registry.agent_service = agent_service
    with patch("app.dependencies.services.get_services", return_value=registry):
        yield agent_service


# --- token masking --------------------------------------------------------


def test_mask_token_keeps_the_public_bot_id_and_hides_the_secret():
    masked = mask_token(BOT_TOKEN)

    assert masked.startswith("123456789:")
    assert "AAExampleTokenValue" not in masked
    assert masked.endswith(BOT_TOKEN[-4:])


def test_mask_token_hides_everything_when_the_shape_is_unexpected():
    assert "•" in mask_token("not-a-real-token")


# --- connect --------------------------------------------------------------


async def test_connect_validates_the_token_and_registers_a_webhook(
    service, telegram_client, org_id
):
    res = await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    assert res.connected is True
    assert res.bot_username == "kita_test_bot"
    telegram_client.get_me.assert_awaited_once_with(BOT_TOKEN)

    _, url, secret = telegram_client.set_webhook.await_args.args
    assert url.startswith("https://kita.test/webhook/telegram/")
    assert secret


async def test_connect_never_returns_the_raw_token(service, org_id):
    res = await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    assert BOT_TOKEN not in (res.masked_token or "")
    assert res.masked_token == mask_token(BOT_TOKEN)


async def test_connect_stores_the_token_encrypted(service, org_id):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    doc = service.integrations.find_one({"org_id": org_id})
    assert doc["bot_token_encrypted"] != BOT_TOKEN
    assert service._decrypt_token(doc) == BOT_TOKEN


async def test_connect_rejects_a_token_telegram_does_not_recognise(
    service, telegram_client, org_id
):
    telegram_client.get_me.side_effect = TelegramApiError("Unauthorized", 401)

    with pytest.raises(IntegrationCredentialError):
        await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    assert service.integrations.find_one({"org_id": org_id}) is None


async def test_connect_fails_before_calling_telegram_when_no_public_url_is_set(
    service, telegram_client, org_id, monkeypatch
):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_BASE_URL", "")

    with pytest.raises(KitaValidationError):
        await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    telegram_client.get_me.assert_not_awaited()


async def test_reconnecting_keeps_the_existing_webhook_url(
    service, org_id, telegram_client
):
    first = await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    rotated = "987654321:BBAnotherTokenValueLongEnough"
    telegram_client.get_me.return_value = {
        "id": 987654321,
        "username": "kita_test_bot",
        "first_name": "Kita Test",
    }
    second = await service.connect(org_id, TelegramConnectRequest(bot_token=rotated))

    assert second.webhook_url == first.webhook_url
    assert service.integrations.count_documents({"org_id": org_id}) == 1


# --- webhook resolution ---------------------------------------------------


async def test_resolve_webhook_accepts_the_matching_secret(service, org_id):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    doc = service.integrations.find_one({"org_id": org_id})

    resolved = service.resolve_webhook(doc["webhook_id"], doc["webhook_secret"])

    assert resolved is not None
    assert resolved["org_id"] == org_id


async def test_resolve_webhook_rejects_a_wrong_secret(service, org_id):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    doc = service.integrations.find_one({"org_id": org_id})

    assert service.resolve_webhook(doc["webhook_id"], "not-the-secret") is None
    assert service.resolve_webhook(doc["webhook_id"], None) is None


def test_resolve_webhook_rejects_an_unknown_webhook_id(service):
    assert service.resolve_webhook("nope", "whatever") is None


# --- processing updates ---------------------------------------------------


async def test_an_inbound_message_creates_a_thread_and_an_agent_reply(
    service, org_id, telegram_client, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})

    await service.process_update(integration, update_payload(1, "hi there"))

    threads = service.list_threads(org_id)
    assert len(threads) == 1
    assert threads[0].telegram_chat_id == 555

    messages = service.list_messages(org_id, threads[0].id)
    assert [(m.sender, m.text) for m in messages] == [
        (TelegramSender.USER, "hi there"),
        (TelegramSender.AGENT, "Hello from the agent"),
    ]
    telegram_client.send_message.assert_awaited_once_with(
        BOT_TOKEN, 555, "Hello from the agent"
    )


async def test_a_redelivered_update_is_processed_only_once(
    service, org_id, telegram_client, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})

    await service.process_update(integration, update_payload(7, "hi there"))
    await service.process_update(integration, update_payload(7, "hi there"))

    assert telegram_client.send_message.await_count == 1
    thread = service.list_threads(org_id)[0]
    assert len(service.list_messages(org_id, thread.id)) == 2


async def test_a_second_message_reuses_the_same_thread(
    service, org_id, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})

    await service.process_update(integration, update_payload(1, "first"))
    await service.process_update(integration, update_payload(2, "second"))

    assert len(service.list_threads(org_id)) == 1


async def test_agent_history_is_carried_into_the_next_run(
    service, org_id, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})

    await service.process_update(integration, update_payload(1, "first"))
    await service.process_update(integration, update_payload(2, "second"))

    # The first run has no history to replay; the second is handed whatever
    # the first stored, which is what gives a Telegram conversation memory.
    assert patched_registry.run.await_args_list[0].kwargs["message_history"] is None
    assert "message_history" in patched_registry.run.await_args_list[1].kwargs


async def test_a_non_text_update_is_ignored(service, org_id, telegram_client):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})

    await service.process_update(
        integration,
        {
            "update_id": 3,
            "message": {
                "message_id": 30,
                "chat": {"id": 555, "type": "private"},
                "sticker": {"file_id": "abc"},
            },
        },
    )

    assert service.list_threads(org_id) == []
    telegram_client.send_message.assert_not_awaited()


async def test_auto_reply_off_on_the_integration_leaves_the_message_for_a_human(
    service, org_id, telegram_client, patched_registry
):
    await service.connect(
        org_id, TelegramConnectRequest(bot_token=BOT_TOKEN, auto_reply=False)
    )
    integration = service.integrations.find_one({"org_id": org_id})

    await service.process_update(integration, update_payload(1, "hi there"))

    telegram_client.send_message.assert_not_awaited()
    thread = service.list_threads(org_id)[0]
    assert thread.unread_count == 1
    assert [m.sender for m in service.list_messages(org_id, thread.id)] == [
        TelegramSender.USER
    ]


async def test_auto_reply_off_on_one_thread_does_not_silence_the_others(
    service, org_id, telegram_client, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})

    await service.process_update(integration, update_payload(1, "hi", chat_id=555))
    taken_over = service.list_threads(org_id)[0]
    service.update_thread(org_id, taken_over.id, TelegramThreadUpdate(auto_reply=False))
    telegram_client.send_message.reset_mock()

    await service.process_update(integration, update_payload(2, "again", chat_id=555))
    await service.process_update(integration, update_payload(3, "hello", chat_id=777))

    # Only the untouched conversation got an automated answer.
    assert telegram_client.send_message.await_count == 1
    assert telegram_client.send_message.await_args.args[1] == 777


async def test_a_failing_agent_does_not_lose_the_inbound_message(
    service, org_id, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})
    patched_registry.run.side_effect = RuntimeError("the model fell over")

    await service.process_update(integration, update_payload(1, "hi there"))

    thread = service.list_threads(org_id)[0]
    assert [m.text for m in service.list_messages(org_id, thread.id)] == ["hi there"]


async def test_an_org_with_no_agent_still_records_the_message(
    service, org_id, telegram_client, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})
    patched_registry.get_all_agents.return_value = []

    await service.process_update(integration, update_payload(1, "hi there"))

    telegram_client.send_message.assert_not_awaited()
    thread = service.list_threads(org_id)[0]
    assert len(service.list_messages(org_id, thread.id)) == 1


# --- tenant isolation -----------------------------------------------------


async def test_one_org_cannot_see_another_orgs_threads(
    service, org_id, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})
    await service.process_update(integration, update_payload(1, "hi there"))

    assert service.list_threads(OTHER_ORG) == []

    thread_id = service.list_threads(org_id)[0].id
    with pytest.raises(TelegramThreadNotFoundError):
        service.list_messages(OTHER_ORG, thread_id)


# --- manual replies -------------------------------------------------------


async def test_a_member_reply_is_sent_and_attributed_to_the_member(
    service, org_id, telegram_client, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})
    await service.process_update(integration, update_payload(1, "hi there"))
    thread = service.list_threads(org_id)[0]

    sent = await service.send_manual_reply(org_id, thread.id, "A human here.")

    assert sent.sender == TelegramSender.MEMBER
    assert telegram_client.send_message.await_args.args[2] == "A human here."


async def test_a_member_reply_is_kept_out_of_the_agents_history(
    service, org_id, patched_registry
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    integration = service.integrations.find_one({"org_id": org_id})
    await service.process_update(integration, update_payload(1, "hi there"))
    thread = service.list_threads(org_id)[0]
    before = service.threads.find_one({"org_id": org_id})["agent_history"]

    await service.send_manual_reply(org_id, thread.id, "A human here.")

    after = service.threads.find_one({"org_id": org_id})["agent_history"]
    assert after == before


async def test_a_manual_reply_needs_a_connected_integration(service, org_id):
    with pytest.raises(IntegrationNotConfiguredError):
        await service.send_manual_reply(org_id, "64b7f1c2e4b0a1a2b3c4d5e6", "hi")


async def test_marking_a_thread_read_clears_its_unread_count(
    service, org_id, patched_registry
):
    await service.connect(
        org_id, TelegramConnectRequest(bot_token=BOT_TOKEN, auto_reply=False)
    )
    integration = service.integrations.find_one({"org_id": org_id})
    await service.process_update(integration, update_payload(1, "hi there"))
    thread = service.list_threads(org_id)[0]
    assert thread.unread_count == 1

    assert service.mark_read(org_id, thread.id).unread_count == 0


# --- configuration lifecycle ---------------------------------------------


async def test_get_integration_reports_a_disconnected_org(service, org_id):
    res = service.get_integration(org_id)

    assert res.connected is False
    assert res.masked_token is None


async def test_updating_the_integration_rebinds_the_agent(service, org_id):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    res = service.update_integration(
        org_id, TelegramIntegrationUpdate(agent_id="agent_other", auto_reply=False)
    )

    assert res.agent_id == "agent_other"
    assert res.auto_reply is False


def test_updating_an_unconnected_integration_is_a_404(service, org_id):
    with pytest.raises(IntegrationNotConfiguredError):
        service.update_integration(org_id, TelegramIntegrationUpdate(auto_reply=False))


async def test_disconnect_removes_the_webhook_and_the_record(
    service, org_id, telegram_client
):
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))

    await service.disconnect(org_id)

    telegram_client.delete_webhook.assert_awaited_once_with(BOT_TOKEN)
    assert service.get_integration(org_id).connected is False


async def test_disconnect_succeeds_even_when_telegram_rejects_the_token(
    service, org_id, telegram_client
):
    """A revoked token must not strand the org with an integration it cannot
    remove and therefore cannot replace."""
    await service.connect(org_id, TelegramConnectRequest(bot_token=BOT_TOKEN))
    telegram_client.delete_webhook.side_effect = TelegramApiError("Unauthorized", 401)

    await service.disconnect(org_id)

    assert service.get_integration(org_id).connected is False
