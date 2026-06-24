# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

# Chat Context Management

## Archival
- `ChatArchiveService` archives old messages when turn count exceeds `CHAT_ARCHIVE_TRIGGER` (default 40)
- Keeps last `CHAT_ARCHIVE_THRESHOLD` (default 20) messages active
- Archived chunks get an LLM summary via `CONTEXT_LLM_MODEL` (default `google/gemini-2.0-flash-001`)
- Archives stored in `chat_archives` collection
- Chat document stores `summary` (merged archive summaries) and `message_count`

## KV Fact Store
- `ChatContextService` uses MongoDB `chat_contexts` collection + Redis hash cache (`chat_context:{chat_id}`) with 1h TTL
- Facts extracted by `__system__context_agent` using `remember_fact` tool
- Context agent uses a small LLM (`CONTEXT_LLM_MODEL`) and is created during org scaffolding

## Context Injection
Before each chat continuation, `_inject_context()` prepends to message_history:
1. Archived summary as `SystemPromptMessage`
2. KV facts as `SystemPromptMessage`

## System Agents
Agents with names starting with `__system__` are filtered from user-facing endpoints (`get_agent`, `get_all_agents`).
