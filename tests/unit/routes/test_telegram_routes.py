"""Route-level behaviour for the Telegram webhook and the integration API."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.dependencies import get_telegram_service
from app.exceptions import IntegrationNotConfiguredError, TelegramThreadNotFoundError
from app.models.telegram import (
    TelegramDirection,
    TelegramIntegrationResponse,
    TelegramMessageResponse,
    TelegramSender,
    TelegramThreadResponse,
)
from app.routes import integration as integration_routes
from app.routes.webhook import telegram as telegram_webhook
from app.security import require_org_membership
from tests.unit.routes.conftest import override

ORG_ID = "org_test_0001"
THREAD_ID = "64b7f1c2e4b0a1a2b3c4d5e6"


def a_thread() -> TelegramThreadResponse:
    now = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    return TelegramThreadResponse(
        id=THREAD_ID,
        telegram_chat_id=555,
        display_name="Ada",
        created_at=now,
        updated_at=now,
    )


def a_message() -> TelegramMessageResponse:
    return TelegramMessageResponse(
        id="64b7f1c2e4b0a1a2b3c4d5e7",
        thread_id=THREAD_ID,
        direction=TelegramDirection.OUTBOUND,
        sender=TelegramSender.MEMBER,
        text="A human here.",
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def telegram_service() -> MagicMock:
    service = MagicMock(name="telegram_service")
    service.resolve_webhook.return_value = {"_id": "1", "org_id": ORG_ID}
    service.process_update = AsyncMock()
    service.get_integration.return_value = TelegramIntegrationResponse(connected=False)
    service.connect = AsyncMock(
        return_value=TelegramIntegrationResponse(
            connected=True, bot_username="kita_test_bot", masked_token="123:••••abcd"
        )
    )
    service.update_integration.return_value = TelegramIntegrationResponse(
        connected=True, bot_username="kita_test_bot", auto_reply=False
    )
    service.disconnect = AsyncMock()
    service.list_threads.return_value = [a_thread()]
    service.get_thread.return_value = a_thread()
    service.update_thread.return_value = a_thread()
    service.mark_read.return_value = a_thread()
    service.list_messages.return_value = [a_message()]
    service.send_manual_reply = AsyncMock(return_value=a_message())
    return service


@pytest.fixture
def webhook_client(make_client, telegram_service):
    return make_client(
        telegram_webhook.router,
        {get_telegram_service: override(telegram_service)},
    )


@pytest.fixture
def api_client(make_client, telegram_service):
    return make_client(
        integration_routes.router,
        {
            get_telegram_service: override(telegram_service),
            require_org_membership: override(ORG_ID),
        },
    )


# --- webhook --------------------------------------------------------------


def test_a_valid_delivery_is_acknowledged_immediately(webhook_client, telegram_service):
    res = webhook_client.post(
        "/webhook/telegram/wh_abc",
        json={"update_id": 1},
        headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"},
    )

    assert res.status_code == 200
    assert res.json() == {"ok": True}
    telegram_service.resolve_webhook.assert_called_once_with("wh_abc", "s3cret")


def test_the_agent_run_is_deferred_rather_than_awaited_inline(
    webhook_client, telegram_service
):
    """Telegram redelivers anything it is not acknowledged for quickly, so the
    handler must not block on the reply."""
    webhook_client.post(
        "/webhook/telegram/wh_abc",
        json={"update_id": 1},
        headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"},
    )

    telegram_service.process_update.assert_awaited_once()


def test_a_delivery_without_the_secret_header_is_refused(
    webhook_client, telegram_service
):
    telegram_service.resolve_webhook.return_value = None

    res = webhook_client.post("/webhook/telegram/wh_abc", json={"update_id": 1})

    assert res.status_code == 403
    telegram_service.process_update.assert_not_awaited()


def test_an_unknown_webhook_id_is_refused_the_same_way_as_a_bad_secret(
    webhook_client, telegram_service
):
    telegram_service.resolve_webhook.return_value = None

    res = webhook_client.post(
        "/webhook/telegram/unknown",
        json={"update_id": 1},
        headers={"X-Telegram-Bot-Api-Secret-Token": "whatever"},
    )

    assert res.status_code == 403


def test_a_malformed_body_is_rejected_without_scheduling_work(
    webhook_client, telegram_service
):
    res = webhook_client.post(
        "/webhook/telegram/wh_abc",
        content=b"not json",
        headers={
            "X-Telegram-Bot-Api-Secret-Token": "s3cret",
            "Content-Type": "application/json",
        },
    )

    assert res.status_code == 400
    telegram_service.process_update.assert_not_awaited()


# --- integration management ----------------------------------------------


def test_get_returns_the_orgs_integration_status(api_client, telegram_service):
    res = api_client.get("/integrations/telegram/")

    assert res.status_code == 200
    assert res.json()["connected"] is False
    telegram_service.get_integration.assert_called_once_with(ORG_ID)


def test_connect_passes_the_token_through_and_returns_it_masked(
    api_client, telegram_service
):
    res = api_client.post(
        "/integrations/telegram/connect",
        json={"bot_token": "123456789:AAExampleTokenValueThatIsLongEnough"},
    )

    assert res.status_code == 200
    assert res.json()["masked_token"] == "123:••••abcd"
    assert telegram_service.connect.await_args.args[0] == ORG_ID


def test_connect_rejects_an_obviously_malformed_token(api_client, telegram_service):
    res = api_client.post("/integrations/telegram/connect", json={"bot_token": "short"})

    assert res.status_code == 422
    telegram_service.connect.assert_not_awaited()


def test_patch_updates_the_integration(api_client, telegram_service):
    res = api_client.patch("/integrations/telegram/", json={"auto_reply": False})

    assert res.status_code == 200
    telegram_service.update_integration.assert_called_once()


def test_delete_disconnects_the_integration(api_client, telegram_service):
    res = api_client.request("DELETE", "/integrations/telegram/")

    assert res.status_code == 200
    assert res.json() == {"status": "disconnected"}
    telegram_service.disconnect.assert_awaited_once_with(ORG_ID)


def test_deleting_an_unconnected_integration_is_a_404(api_client, telegram_service):
    telegram_service.disconnect.side_effect = IntegrationNotConfiguredError("telegram")

    res = api_client.request("DELETE", "/integrations/telegram/")

    assert res.status_code == 404


# --- inbox ----------------------------------------------------------------


def test_threads_are_listed_for_the_authenticated_org(api_client, telegram_service):
    res = api_client.get("/integrations/telegram/threads")

    assert res.status_code == 200
    assert res.json()[0]["display_name"] == "Ada"
    telegram_service.list_threads.assert_called_once_with(ORG_ID, limit=50)


def test_thread_messages_are_listed(api_client, telegram_service):
    res = api_client.get(f"/integrations/telegram/threads/{THREAD_ID}/messages")

    assert res.status_code == 200
    assert res.json()[0]["text"] == "A human here."


def test_an_unknown_thread_is_a_404(api_client, telegram_service):
    telegram_service.list_messages.side_effect = TelegramThreadNotFoundError("nope")

    res = api_client.get("/integrations/telegram/threads/nope/messages")

    assert res.status_code == 404


def test_sending_a_reply_reaches_the_service(api_client, telegram_service):
    res = api_client.post(
        f"/integrations/telegram/threads/{THREAD_ID}/messages",
        json={"text": "A human here."},
    )

    assert res.status_code == 200
    telegram_service.send_manual_reply.assert_awaited_once_with(
        ORG_ID, THREAD_ID, "A human here."
    )


def test_an_empty_reply_is_rejected(api_client, telegram_service):
    res = api_client.post(
        f"/integrations/telegram/threads/{THREAD_ID}/messages", json={"text": ""}
    )

    assert res.status_code == 422
    telegram_service.send_manual_reply.assert_not_awaited()


def test_a_reply_longer_than_telegram_allows_is_rejected(api_client, telegram_service):
    res = api_client.post(
        f"/integrations/telegram/threads/{THREAD_ID}/messages",
        json={"text": "x" * 4097},
    )

    assert res.status_code == 422


def test_toggling_thread_auto_reply_reaches_the_service(api_client, telegram_service):
    res = api_client.patch(
        f"/integrations/telegram/threads/{THREAD_ID}", json={"auto_reply": False}
    )

    assert res.status_code == 200
    args = telegram_service.update_thread.call_args.args
    assert args[0] == ORG_ID and args[1] == THREAD_ID
    assert args[2].auto_reply is False


def test_marking_a_thread_read_reaches_the_service(api_client, telegram_service):
    res = api_client.post(f"/integrations/telegram/threads/{THREAD_ID}/read")

    assert res.status_code == 200
    telegram_service.mark_read.assert_called_once_with(ORG_ID, THREAD_ID)
