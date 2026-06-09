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
