# Telegram Integration — Design

How an organization connects a Telegram bot to Kita, so people on Telegram can
talk to the org's agent, and so org members can read and take over those
conversations from the Kita UI.

## Shape of the feature

```
Telegram user ──DM──▶ Telegram ──webhook──▶ POST /webhook/telegram/{webhook_id}
                                                     │
                                            resolve org by webhook_id
                                            verify X-Telegram-Bot-Api-Secret-Token
                                            dedupe by update_id
                                                     │
                                            persist inbound message
                                                     │
                                     auto_reply on? ──yes──▶ agent.run(history)
                                                     │              │
                                                     no      sendMessage back
                                                     │
                                       thread waits for a human in
                                       Kita UI ▸ Inbox ▸ reply box
```

An org member in the Kita UI sees every thread, can reply as the bot, and can
flip auto-reply off per thread to take over ("human handoff").

## Design choices

Each entry is the decision, the alternative that lost, and why.

### 1. One bot per organization (BYO token), not one shared Kita bot

**Chosen:** each org registers its own bot token from `@BotFather`.

The alternative — a single Kita-owned bot that routes by `/start <org_code>`
deep links — is easier to onboard but gives every customer the same bot
identity and username, and makes routing depend on the user having gone
through a deep link rather than on the message itself. A per-org bot is also
what the existing Facebook integration already assumes (`facebook_page_id` on
the org), so this keeps one mental model per channel.

Cost: the org admin has to create a bot before connecting. Accepted.

### 2. Webhook transport, with an explicit tunnel story for local dev

**Chosen:** `setWebhook` at connect time; Telegram pushes updates to us.

Long-polling (`getUpdates`) would work without a public URL, but it needs a
process per bot holding an open connection, which does not fit a stateless
FastAPI deployment and would need the Hatchet worker to babysit it. Webhooks
cost nothing when idle and match the Facebook route already in the repo.

The consequence is that `localhost` cannot receive updates. For local E2E,
point `TELEGRAM_WEBHOOK_BASE_URL` at an `ngrok http 8000` tunnel; the connect
call reads that env var when building the webhook URL.

### 3. Opaque per-org webhook path + Telegram's secret token header

**Chosen:** the webhook URL is `/webhook/telegram/{webhook_id}` where
`webhook_id` is a random 32-char token generated per org, and `setWebhook` is
called with a separate random `secret_token` that Telegram echoes back in the
`X-Telegram-Bot-Api-Secret-Token` header on every request.

Telegram does not sign payloads the way Facebook does with
`X-Hub-Signature-256`, so identity has to come from the URL or a shared
secret. The common shortcut is to put the bot token itself in the path
(`/webhook/<bot_token>`); that leaks a full-privilege credential into every
proxy log and access log it passes through. Two independent random values —
one routing, one authenticating — cost nothing extra and neither is the bot
token.

`webhook_id` is indexed, so routing is one lookup and does not require
parsing the bot id out of the payload.

### 4. Bot token encrypted at rest, never returned by the API

**Chosen:** store the token through the existing `app/services/encryption.py`
Fernet helpers, and return only a mask (`1234…wxyz`) in API responses.

The bot token is a full-privilege credential: it can read every message the
bot receives and post as the org. It sits in the same risk class as the API
keys that already go through `encrypt_key`, so it gets the same treatment
rather than a new one.

### 5. Dedicated collections, not a field on `Integrations`

**Chosen:** `telegram_integrations`, `telegram_threads`, `telegram_messages`.

The org's `Integrations` model is a flat bag of ids (`facebook_page_id`), and
Telegram needs real state: encrypted token, bot metadata, webhook id, secret,
bound agent, auto-reply default, connection status. Cramming that into the
org document would bloat every org read and make the token available anywhere
an org is loaded. Separate collections keep the credential's blast radius
small and let threads and messages be indexed and paged properly.

All three are read through `TenantCollection`, so `org_id` is injected on
every query — except the webhook lookup, which is by definition
pre-authentication and queries the raw collection by `webhook_id`.

### 6. Conversation history lives with the thread, agent history is rebuilt

**Chosen:** `telegram_messages` stores the human-readable transcript
(`direction`, `text`, `telegram_message_id`, timestamps), and the pydantic-ai
message history for the agent is persisted per thread in `agent_history`.

Reusing the `chats` collection was tempting — the agent's history format
already lives there — but a `chats` document is scoped to a UI chat session
with an `agent_id` and drives the chat list UI. Overloading it with Telegram
threads would mean every chat list query has to filter out a channel it does
not care about, and the Telegram inbox would have to reverse-engineer a
transcript out of pydantic-ai part arrays for display. Keeping a plain
transcript for humans and an opaque history blob for the agent means neither
side has to translate.

### 7. Acknowledge fast, answer in the background

**Chosen:** the webhook validates, persists, returns `200` immediately, and
runs the agent in a FastAPI `BackgroundTask`.

Telegram retries an update that is not acknowledged quickly, and an agent run
can take tens of seconds — replying inline would guarantee duplicate
deliveries and duplicate replies. Hatchet is available and is the right home
for this at volume, but a background task has no extra moving parts and the
work is already idempotent thanks to update deduplication. Noted as the
scale-out path rather than built now.

### 8. `update_id` deduplication

Telegram redelivers any update it did not get a `200` for, and a background
task means we ack before the work is done. Every processed `update_id` is
recorded per integration with a unique index; a repeat is dropped before any
agent run. Without this, one slow reply becomes several identical replies.

### 9. Auto-reply is a switch at two levels

The integration has a default (`auto_reply`), and each thread can override it.
Turning it off for one thread is how a human takes over a conversation without
disconnecting the bot for everyone else.

### 10. UI: separate Integrations and Inbox pages, polled not pushed

**Chosen:** `/integrations` for connect/disconnect/configure, `/inbox` for
conversations; both fetch through TanStack Query, with the inbox polling on an
interval.

The repo already has a chat status WebSocket, so pushing was an option, but it
is scoped to a single agent run rather than to a stream of inbound messages,
and wiring a second socket for a feature whose traffic is human-typing-speed
is not worth the reconnection and fan-out handling. A poll is a few lines and
is trivially correct. Swap it for a socket when the inbox is busy enough to
notice.

## Data model

`telegram_integrations` (one per org)

| field | notes |
|---|---|
| `org_id` | tenant key |
| `bot_token_encrypted` | Fernet ciphertext |
| `bot_id`, `bot_username`, `bot_name` | from `getMe` at connect time |
| `webhook_id` | random, unique, indexed — routes the webhook |
| `webhook_secret` | random, compared against the Telegram header |
| `agent_id` | which agent answers; defaults to the org's first agent |
| `auto_reply` | org-level default |
| `status` | `connected` / `error` |

`telegram_threads` (one per Telegram chat)

| field | notes |
|---|---|
| `org_id`, `telegram_chat_id` | unique together |
| `chat_type`, `title`, `username`, `first_name`, `last_name` | display |
| `auto_reply` | `None` = inherit the integration default |
| `last_message_at`, `last_message_preview`, `unread_count` | inbox list |
| `agent_history` | pydantic-ai history blob for the next run |

`telegram_messages`

| field | notes |
|---|---|
| `org_id`, `thread_id` | tenant + thread |
| `direction` | `inbound` / `outbound` |
| `sender` | `user` / `agent` / `member` |
| `text`, `telegram_message_id`, `created_at` | |

## API surface

Public (no org auth — authenticated by the secret header):

- `POST /webhook/telegram/{webhook_id}`

Org-scoped (`require_org_membership`):

- `GET    /integrations/telegram` — status, masked token, bot info
- `POST   /integrations/telegram/connect` — validate token, set webhook, store
- `PATCH  /integrations/telegram` — bind agent, toggle org-level auto-reply
- `DELETE /integrations/telegram` — delete webhook upstream, drop the record
- `GET    /integrations/telegram/threads`
- `GET    /integrations/telegram/threads/{thread_id}/messages`
- `POST   /integrations/telegram/threads/{thread_id}/messages` — reply as a member
- `PATCH  /integrations/telegram/threads/{thread_id}` — per-thread auto-reply

## Environment

```
TELEGRAM_WEBHOOK_BASE_URL=https://<public-host>   # ngrok tunnel in local dev
API_KEY_ENCRYPTION_KEY=<fernet key>               # already required
```

Both live in Doppler (`kita-api` / `dev`), so the API is started through it:

```
doppler run -- uv run python main.py
```

**`TELEGRAM_WEBHOOK_BASE_URL` is read when a bot is connected, not when a
message arrives.** Connecting bakes the value into the URL handed to
`setWebhook`, and Telegram keeps delivering there until something re-registers
it. So a server still holding a stale value from its environment will happily
register a webhook pointing somewhere the code does not run, and the only
symptom is silence in the chat — the API sees nothing at all. Ask Telegram what
it thinks rather than reading local logs:

```
curl -s "https://api.telegram.org/bot<token>/getWebhookInfo" | jq
```

`url` is where deliveries are actually going, and `last_error_message` names
the failure — `404 Not Found` means it is pointed at something without these
routes. `pending_update_count` is the backlog waiting on a fix.

Because the environment is read at connect time, changing it in Doppler is not
enough on its own: restart the API before reconnecting, or the old value is
what gets registered.

## Local E2E

1. Create a bot with `@BotFather`, keep the token.
2. `ngrok http 8000` and set `TELEGRAM_WEBHOOK_BASE_URL` in Doppler to the
   https URL it prints. A free ngrok URL changes every restart, so this is a
   per-session step, not a one-off.
3. Start the API through `doppler run --` *after* setting it, and start the UI.
4. Connect the bot on `/integrations`.
5. Message the bot from a real Telegram account; the reply should arrive in
   Telegram and the thread should appear in `/inbox`.

If step 5 is silent, run `getWebhookInfo` before anything else — it
distinguishes "never reached us" from "reached us and failed" in one call.
