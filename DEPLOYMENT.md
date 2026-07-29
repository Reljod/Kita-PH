# 🚀 Production Deployment Guide

This guide details the procedures, environment configurations, and deployment strategies required to run `kita-api` in staging and production environments.

---

## 📋 Infrastructure Requirements

To run `kita-api` in production, ensure you have provisioned and configured the following services:

1. **MongoDB**: A highly available cluster (such as [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas-database)) with appropriate access controls.
2. **Redis**: A low-latency Redis cache (such as Redis Enterprise or AWS ElastiCache) for token caching and API rate limiting.
3. **LLM Orchestration**: Access credentials to OpenRouter (or your chosen model endpoint) and Pydantic AI integrations.
4. **Graph Database**: Neo4j instance (such as [Neo4j Aura](https://neo4j.com/product/auradb/)).
5. **Worker Queue**: Hatchet Cloud or self-hosted Hatchet instance credentials.
6. **Observability**: Logfire token for distributed tracing and performance metrics.

---

## ⚙️ Environment Variables

Below is the exhaustive checklist of environment variables that **must** be set in your production/staging environment. 

> [!IMPORTANT]
> Never commit secrets or actual production configuration values to source control. Use your hosting provider's secure secret manager (e.g., AWS Secrets Manager, Vercel/Render Environment Variables, or Heroku Config Vars).

| Variable | Scope / Type | Example / Format | Description |
| :--- | :--- | :--- | :--- |
| `MONGO_URI` | **Secret** | `mongodb+srv://<user>:<password>@cluster.mongodb.net/...` | Production MongoDB connection string. |
| `MONGO_DB_NAME` | Config | `kita_prod` | The production MongoDB database name. |
| `REDIS_CONNECTION_STRING` | **Secret** | `redis://:<password>@redis-host:port/0` | Production Redis connection string. |
| `API_KEY_ENCRYPTION_KEY` | **Secret** | `32-byte Fernet key string` | Used by the middleware to decrypt API keys. **Must** match the key used when generating client keys in development. |
| `OPENROUTER_API_KEY` | **Secret** | `sk-or-v1-...` | Token used to authenticate model calls via OpenRouter. |
| `CORS_ALLOWED_ORIGINS` | Config | `https://kita-agents.dev,https://app.kita-agents.dev` | Comma-separated list of web app URLs authorized to make cross-origin requests. |
| `LOGFIRE_TOKEN` | **Secret** | `lf_tok_...` | Logfire instrumentation telemetry token. |
| `SUPABASE_URL` | Config | `https://your-project.supabase.co` | Supabase storage API URL. |
| `SUPABASE_KEY` | **Secret** | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` | Supabase service role or API key. |
| `HATCHET_CLIENT_TOKEN` | **Secret** | `hatchet_cli_tok_...` | Hatchet worker client token. |
| `NEO4J_URI` | Config | `neo4j+s://<db-id>.databases.neo4j.io` | Neo4j Aura host endpoint. |
| `NEO4J_USERNAME` | Config | `neo4j` | Username for the graph database. |
| `NEO4J_PASSWORD` | **Secret** | `<strong-password>` | Authentication secret for the graph database. |

---

## 🤖 Continuous Deployment (FastAPI Cloud)

`.github/workflows/deploy.yml` ships `main` to FastAPI Cloud on every merge.
The run is `Tests → FastAPI Cloud`: the deploy job only starts once the full
suite (and the 90% coverage gate) has passed on the merge commit itself, and
production deploys are serialized so two merges can never race to promote.

### One-time setup

Everything lives on the **`Production – kita-ph`** GitHub Actions environment
(*Settings → Environments*), so deploy credentials are never exposed to a
workflow running on a fork or a feature branch:

| Name | Kind | Value |
| :--- | :--- | :--- |
| `FASTAPI_CLOUD_TOKEN` | **Secret** | A FastAPI Cloud deploy token (`fcd_...`). |
| `FASTAPI_CLOUD_APP_ID` | Variable | The app UUID from the dashboard header. A secret of the same name also works. |
| `FASTAPI_CLOUD_APP_URL` | Variable *(optional)* | Public app URL, e.g. `https://kita-api.fastapicloud.dev`. Enables the post-deploy smoke check. |

Add required reviewers to that environment if you want a merge to pause for
manual approval before it ships — the workflow needs no changes for that.

### What the pipeline does and does not cover

- **Does not set application environment variables.** A deploy token is scoped
  to deploying; `fastapi cloud env set` and `fastapi cloud logs` both return
  `401` under one. The variables in the table above are for *GitHub*; the
  application's own variables are a dashboard action. Change them there, then
  re-run the workflow (**Actions → Deploy → Run workflow**) so the running
  app picks them up.
- **Fails loudly on a bad build.** The CLI streams the build log into the job
  and exits non-zero if the build or the promotion fails.
- **Verifies the app actually serves**, when `FASTAPI_CLOUD_APP_URL` is set, by
  polling `/openapi.json` — a route that needs neither auth nor a database, so
  a `200` isolates "the new build booted" from "the database is misconfigured".
  The probe retries with a widening backoff on purpose: the app can scale to
  zero, and a cold first request may take tens of seconds before it answers.

### ⏪ Rolling back

There is no "promote the previous deployment" button available to CI — a deploy
token can only *deploy*. So a rollback is a roll-forward: re-deploy an older
commit with the same workflow.

`workflow_dispatch` runs a **branch or tag**, never a bare commit SHA, so tag
the last good commit first. Find it under **Actions → Deploy** — the newest run
with a green *FastAPI Cloud* job.

```bash
# 1. Point a tag at the last commit that deployed cleanly.
git tag rollback-2026-07-28 <good-sha>
git push origin rollback-2026-07-28

# 2. Ship that ref, then watch it land.
gh workflow run deploy.yml --ref rollback-2026-07-28
gh run watch "$(gh run list --workflow=deploy.yml --limit 1 \
  --json databaseId --jq '.[0].databaseId')"
```

The same thing from the UI: **Actions → Deploy → Run workflow**, then pick the
tag from the ref dropdown.

Three things to know before you reach for this:

- **The tag must contain `.github/workflows/deploy.yml`.** GitHub reads the
  workflow *from the ref you dispatch*, so anything older than the commit that
  introduced the file cannot be dispatched at all. For those, branch off the old
  commit and cherry-pick the workflow onto it.
- **The full test suite runs again** at that old ref, because `deploy` still
  `needs: test`. That is deliberate — a rollback target that cannot pass its own
  tests is not a safe place to land — but it means a rollback costs a full CI
  run, not seconds.
- **Rolling back the app does not change `main`.** The bad commit is still what
  `main` points at, so the *next* merge redeploys it. Follow up with a
  `git revert` PR; the deploy that merge triggers is what makes the rollback
  permanent.

Manual and local deploys are unchanged: see `.agents/skills/deploy-fastapi-cloud`.

---

## 📦 Container Deployment (Docker)

The project includes a multi-purpose `Dockerfile` optimized for minimal resource footprints.

### ⚠️ Critical Step: Generate `requirements.txt`
The `Dockerfile` relies on a standard `requirements.txt` dependency file. Since dependencies are locally managed via `uv.lock`, you **must** generate the `requirements.txt` file before triggering a Docker build.

Run the following command in your build environment:
```bash
# Export uv lockfile to standard pip requirements format
uv pip export -o requirements.txt
```
*Alternatively, you can use:*
```bash
uv export --format requirements.txt > requirements.txt
```

### Build & Run Commands
Once `requirements.txt` is present, execute the following commands in the project root:

1. **Build the Docker Image**:
   ```bash
   docker build -t kita-api:latest .
   ```

2. **Run the Docker Container Locally**:
   ```bash
   docker run -d \
     -p 8000:8000 \
     --env-file .env.local \
     --name kita-api-instance \
     kita-api:latest
   ```

3. **Verify the Deployment**:
   ```bash
   curl -I http://localhost:8000/
   ```
   *Expected Response:* HTTP/1.1 200 OK (if headers bypass or requests succeed).

---

## ☁️ PaaS Deployment (Render / Heroku / Dokku)

Kita API contains a `Procfile` in the root directory, making it natively compatible with PaaS hosting solutions like Render, Heroku, and Dokku.

The `Procfile` specifies the web dyno command:
```web
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Configuration Steps:
1. **Connect Repository**: Link your GitHub repository to your PaaS application.
2. **Set Build Command**:
   Configure the build step to install `uv` and export dependencies, or use the Docker-based deployment option provided by Render/Heroku (which automatically builds using the repository's `Dockerfile`).
   - If using the **Docker-based builder**, set up a pre-build hook to run `uv pip export -o requirements.txt` or configure the deployment pipeline to run the export before building the image.
   - If using **Python buildpacks**, configure the build command to generate `requirements.txt` so the standard Python buildpack can locate and install dependencies:
     ```bash
     pip install uv && uv pip export -o requirements.txt && pip install -r requirements.txt
     ```
3. **Environment Setup**: Define all required keys listed in the [Environment Variables](#-environment-variables) section in the dashboard of your PaaS platform.
4. **Port Configuration**: PaaS environments automatically set the `$PORT` environment variable. The `Procfile` and `Dockerfile` are already configured to bind to this dynamic port.
