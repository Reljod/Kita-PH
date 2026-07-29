# AGENTS.md — Kita API

This file is the operating charter for any agent working in this repository —
Claude Code, a Claude Agent SDK process, or any other AGENTS.md-compatible
tool. `CLAUDE.md` is a symlink to this file: one charter, read by every
runtime.

## What this repo is

FastAPI + uv backend powering agent execution, memory sync, and background workflows for the Kita ecosystem.

## Operating principles

1. **Act decisively where the call is clearly yours.** Prefer well-reasoned
   action over hedged options; escalate the genuinely ambiguous decisions
   instead of guessing.
2. **System of record over ad hoc storage.** Keep each kind of information
   where it actually belongs (tasks in the tracker, docs in the docs, code
   in the repo) rather than scattering shadow copies.
3. **Reversible by default.** Local, reversible actions (drafting, editing,
   reading) don't need a check-in. Anything hard to reverse or visible to
   others — sending messages, pushing to shared branches, closing tickets —
   gets confirmed first.
4. **Extend by writing it down.** Capture what proves itself in the smallest
   durable form: a one-line WHY note under **Design choices** below, or — for
   a repeatable, multi-step procedure — a skill under `.agents/skills/`. Ad
   hoc fixes that never get written down don't compound.
5. **Keep this file thin.** This charter holds identity, principles, and slim
   WHY notes. Operational how-to lives in the skill that owns it; deep
   area-specific procedure in its own linked doc. Not here.

## Design choices (the WHYs)

Slim notes on decisions worth not re-litigating, so the reasoning outlives
the session that set it. Add a line when a choice proves itself; distill it,
don't narrate it.

- **Quality by layering, not diligence.** Cheap deterministic checks early
  under mandatory ones later beats relying on remembering to be careful (see
  the layer model below).
- **This sandbox has HTTPS egress only.** `:443` works, so Supabase,
  OpenRouter and the FastAPI Cloud API are reachable; Mongo Atlas `:27017`
  and Redis Cloud are not. A connection timeout to those is the network, not
  the credentials — don't go looking for a bad password.
- **Deploy tokens deploy, nothing else.** Reading or writing an app's
  environment variables and streaming its logs all need broader scope, so a
  config change is a dashboard action, not something to automate around.
  → **`deploy-fastapi-cloud`**
- **Read the API's own status codes to diagnose a deployment.** A fast
  `401 Invalid x-client-id` proves the database was queried; a slow
  `500 Database error` is what a missing `MONGO_URI` actually looks like.
  → **`deploy-fastapi-cloud`**
- **CI installs the deploy CLI, not the app.** The build runs on FastAPI
  Cloud, so a deploy job that syncs the full project spends minutes pulling
  torch to upload a tarball. `uv sync --only-group dev` is enough.
  → **`deploy-fastapi-cloud`**
- **One CI definition of "the tests", called rather than copied.** `test.yml`
  exposes `workflow_call` and the deploy workflow invokes it, so main is
  gated by the exact suite that guards every PR — and a merge commit, which
  no PR check ever ran against, still has to go green before shipping.
- **Agent versions come from the counter, never from a literal.**
  `create_agent` writing `version=1` while leaving the counter un-seeded put
  two documents at the same `(base_id, version)` and made a pinned
  `<base_id>-v1` ambiguous. Anything allocating a version goes through
  `_next_version`.
- **An agent's language is an enum, never free text.** The value selects a
  fixed instruction block under `templates/languages/`, so the prompt never
  carries user-supplied language text and needs no sanitising on that path.
  English deliberately contributes no block, which keeps un-migrated agents'
  prompts byte-identical to what they were before the setting existed.
- **Every write path rebuilds the agent document by hand.** `update_agent`,
  `add_tools` and `remove_tools` each construct a fresh version copy, so a new
  field has to be added to all of them — one that is missed silently resets to
  its default on an unrelated edit, exactly like a dropped version number.
- **A local lint gate has to match CI's scope, not just its command.**
  `pre-commit` sees only staged files; CI lints everything the branch
  touches relative to `main`. A file already clean when it was staged still
  fails CI, which is how a green local commit produced a red PR. `pre-push`
  runs the same commands over the same branch-wide diff.
  → **`setup-git-hooks`**
- _Add your project's WHYs here as they emerge._

## How code quality is enforced (the layer model)

Quality comes from layering cheap deterministic checks early with mandatory
ones later, so nothing depends on remembering to be careful:

1. **Local, fast, skippable** — git hooks catch typos, formatting, and
   malformed commit messages in under a second. Bypassable; never the real
   gate. → **`setup-git-hooks`**
2. **Server-side, slower, mandatory** — branch protection + required CI
   checks + required reviewers. This is where "tests must pass" and coverage
   thresholds actually bite.
3. **Continuous, not per-PR** — heavier suites (E2E, fuzzing) on a schedule;
   they open a ticket, they don't block a merge.
4. **Upstream of code** — a short behavior list before non-trivial work, so
   "did we build the right thing" is answered before the code exists.
   → **`tdd-loop`**

## Branching

Feature branches only, never directly on `main`. Names follow
`claude/<short-description>-<id>` for agent-driven work.

## Commits

```
<type>: <TICKET> <subject>
```

- `type` ∈ feat, fix, bug, chore, docs, refactor, test, perf, ci, build,
  style, revert.
- `TICKET` is the issue key (e.g. `JOD-12`), required except
  for housekeeping types (chore/docs/style/ci).
- Keep the subject imperative and ≤ 72 chars.

Match whatever `setup-git-hooks` installs for this repo. → **`setup-git-hooks`**

## PRs

Draft by default; open one after pushing if no open PR exists for the
branch. Build the body with **`create-pr`** — visuals first, and surface
that the deterministic checks are green so review attention goes to
judgment, not to re-verifying the boring correctness.

## Skills

Reusable skills live under `.agents/skills/<skill-name>/SKILL.md`, each with
a thin slash-command wrapper in `.claude/commands/`. Invoke either the
command or the skill directly. Promote a new skill here once a behavior has
proven itself more than once.
