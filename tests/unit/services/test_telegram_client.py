"""The HTTP edge to the Telegram Bot API, exercised against a mocked transport."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.services.webhook.telegram_client import (
    MAX_MESSAGE_LENGTH,
    TelegramApiError,
    TelegramClient,
    split_message,
)

BOT_TOKEN = "123456789:AAExampleTokenValueThatIsLongEnough"
BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


@pytest.fixture
def client() -> TelegramClient:
    return TelegramClient(timeout=1.0)


# --- message splitting ----------------------------------------------------


def test_a_short_message_is_left_alone():
    assert split_message("hello") == ["hello"]


def test_an_over_long_message_is_split_rather_than_dropped():
    """Telegram rejects anything past 4096 characters outright, so an unsplit
    long answer would be lost entirely rather than truncated."""
    chunks = split_message("x" * (MAX_MESSAGE_LENGTH + 500))

    assert len(chunks) == 2
    assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)


def test_splitting_prefers_a_paragraph_boundary():
    body = "a" * (MAX_MESSAGE_LENGTH - 10) + "\n\n" + "b" * 100

    chunks = split_message(body)

    assert chunks[0] == "a" * (MAX_MESSAGE_LENGTH - 10)
    assert chunks[1] == "b" * 100


def test_splitting_terminates_on_text_with_no_break_at_all():
    chunks = split_message("y" * (MAX_MESSAGE_LENGTH * 3))

    assert len(chunks) == 3
    assert "".join(chunks) == "y" * (MAX_MESSAGE_LENGTH * 3)


# --- API calls ------------------------------------------------------------


@respx.mock
async def test_get_me_returns_the_result_payload(client):
    respx.post(f"{BASE}/getMe").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"id": 42}})
    )

    assert await client.get_me(BOT_TOKEN) == {"id": 42}


@respx.mock
async def test_a_rejected_token_raises_with_telegrams_description(client):
    respx.post(f"{BASE}/getMe").mock(
        return_value=httpx.Response(
            401, json={"ok": False, "error_code": 401, "description": "Unauthorized"}
        )
    )

    with pytest.raises(TelegramApiError) as excinfo:
        await client.get_me(BOT_TOKEN)

    assert excinfo.value.description == "Unauthorized"
    assert excinfo.value.error_code == 401


@respx.mock
async def test_a_non_json_response_becomes_a_telegram_error(client):
    respx.post(f"{BASE}/getMe").mock(return_value=httpx.Response(502, text="<html>"))

    with pytest.raises(TelegramApiError):
        await client.get_me(BOT_TOKEN)


@respx.mock
async def test_set_webhook_sends_the_secret_and_narrows_the_update_types(client):
    route = respx.post(f"{BASE}/setWebhook").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": True})
    )

    await client.set_webhook(
        BOT_TOKEN, "https://kita.test/webhook/telegram/x", "s3cret"
    )

    body = route.calls.last.request.read().decode()
    assert '"secret_token":"s3cret"' in body.replace(" ", "")
    assert "edited_message" in body


@respx.mock
async def test_send_message_posts_the_text_to_the_chat(client):
    route = respx.post(f"{BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})
    )

    result = await client.send_message(BOT_TOKEN, 555, "hello")

    assert result == {"message_id": 7}
    assert route.call_count == 1


@respx.mock
async def test_an_over_long_reply_is_sent_as_several_messages(client):
    route = respx.post(f"{BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})
    )

    await client.send_message(BOT_TOKEN, 555, "z" * (MAX_MESSAGE_LENGTH + 10))

    assert route.call_count == 2
