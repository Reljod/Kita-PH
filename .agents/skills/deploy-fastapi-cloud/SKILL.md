---
name: deploy-fastapi-cloud
description: >
  Use when deploying the Kita API to FastAPI Cloud, when a deploy token or
  app ID is handed over, or when diagnosing a deployment that built but
  will not serve. Triggers on "deploy the API", "push this to FastAPI
  Cloud", "the deployment is 404ing", "set the env vars on the app".
  Covers non-interactive deploys, what a deploy token can and cannot do,
  and how to tell a missing-config failure from a missing-route one.
---

# deploy-fastapi-cloud

## Deploying without a browser

`fastapi deploy` is non-interactive when both values are in the
environment. Requires `fastapi-cloud-cli >= 0.9.0` (a dev dependency of
this repo; `uv run fastapi --help` should list a `deploy` command).

```bash
export FASTAPI_CLOUD_TOKEN='fcd_...'          # never write this into the repo
export FASTAPI_CLOUD_APP_ID='<uuid>'          # the UUID in the dashboard header
uv run fastapi deploy                         # --no-wait to return after upload
```

Secrets cannot ride along by accident: the CLI hard-excludes `.env` and
anything matching `.env.*` from the upload archive, so a deployed app is
configured **only** by what is set on the platform.

## What a deploy token can and cannot do

A deploy token is scoped to deploying. Everything else returns
`401 You don't have permissions for this resource`:

| Command | Works with a deploy token |
| --- | --- |
| `fastapi deploy` | yes |
| `fastapi cloud env list` / `env set` | **no** |
| `fastapi cloud logs` | **no** |

So if the app needs new environment variables, a deploy token is not
enough — someone has to set them in the dashboard. Do not spend time
looking for a CLI flag that works around this; there isn't one, and the
restriction is deliberate.

## Deploying from CI

`.github/workflows/deploy.yml` already does this on every merge to `main`;
credentials live on the `Production – kita-ph` environment. Two things
make that job cheap, and both are easy to get wrong when editing it:

```bash
uv sync --only-group dev                          # not `uv sync`
uv run --no-sync python -m fastapi_cloud_cli deploy
```

The build happens on FastAPI Cloud, so a runner never needs the app's
runtime dependencies — and `uv run fastapi deploy` would drag in torch
(gigabytes) just to upload a tarball. The dev group alone has no `fastapi`
console script, hence `python -m`; the module entrypoint takes the same
arguments, including `FASTAPI_CLOUD_APP_ID` as the env var behind
`--app-id`.

`fastapi cloud setup-ci` will scaffold this from scratch elsewhere — it
mints a 365-day deploy token and writes both secrets through the `gh` CLI.
It always writes repo-level secrets, never environment-scoped ones.

## Linking without the interactive prompt

`fastapi cloud link` has no `--app-id`, but the config it writes is two
fields. Fetch `team_id` from the API and write the file directly:

```bash
curl -sS -H "Authorization: Bearer $FASTAPI_CLOUD_TOKEN" \
  "https://api.fastapicloud.com/api/v1/apps/$FASTAPI_CLOUD_APP_ID"
mkdir -p .fastapicloud
printf '{"app_id":"%s","team_id":"%s"}' "$FASTAPI_CLOUD_APP_ID" "$TEAM_ID" \
  > .fastapicloud/cloud.json
printf '*' > .fastapicloud/.gitignore     # the folder ignores itself
```

## Diagnosing a deployment that will not serve

Poll the app rather than the deployment id — the CLI often prints
"Could not confirm deployment status" while the platform goes on to
promote a *new* deployment id a few minutes later:

```bash
curl -sS -H "Authorization: Bearer $FASTAPI_CLOUD_TOKEN" \
  "https://api.fastapicloud.com/api/v1/apps/$FASTAPI_CLOUD_APP_ID" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['live_deployments'])"
```

An empty `live_deployments` means it never went live. A populated one with
a 404 at `/` usually means the edge is fine and you are looking at a route
that doesn't exist — check `/docs` and `/openapi.json`, which need no
auth and no database.

### Telling "no config" apart from "no client"

Since `logs` is unavailable under a deploy token, read the status codes.
`ApiKeyAuthMiddleware` distinguishes them for you:

| Response | Latency | Means |
| --- | --- | --- |
| `401 Invalid x-client-id or x-api-key` | ~1–2s | Mongo **is** reachable; that client is not registered |
| `500 Database error during authentication` | ~30s | Mongo is not reachable — env vars are missing or wrong |

```bash
curl -sS -w "\nHTTP %{http_code}\n" \
  -H "x-client-id: probe" -H "x-api-key: 0123456789abcdef01234567" \
  https://<app>.fastapicloud.dev/
```

A fast 401 is good news: the app booted and queried the database. Do not
conclude "the env vars are missing" from a 401 — that is the opposite of
what it means.

## Working credentials

`kita_admin.clients` lives on the shared cluster, so a client registered
for any environment authenticates against the deployment too. The UI's own
pair in `.kita-ui.env` (`KITA_CLIENT_ID` / `KITA_API_KEY`) is the quickest
way to get an authenticated request through without minting anything.

Minting a new one with `scripts/generate_client.py` requires direct Mongo
access on `:27017`, which a sandbox or CI runner often does not have even
when HTTPS works fine.
