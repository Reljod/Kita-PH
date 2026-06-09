# 🌌 Kita API

Kita API is a high-performance, production-ready backend framework for LLM-powered services. Built on top of **FastAPI** and optimized with the lightning-fast **`uv`** package manager, it serves as the core agent execution and memory synchronization service for the Kita ecosystem.

---

## ⚡ Core Features

- **🧠 Multi-Agent Execution**: Built-in support for orchestrating complex LLM workloads powered by `pydantic-ai`.
- **📁 Advanced File Uploading**: Efficient upload mechanisms for standard and large files (with standard binary PUT and resumable `TUS` protocol configurations).
- **💾 Semantic memory (RAG)**: Organization-isolated vector storage and retrieval.
- **⚡ Background Workflows**: Event-driven scheduling and long-running task offloading using Hatchet.
- **🔌 Enterprise Integration**: Complete webhook support (e.g., Facebook Messenger, events framework).
- **🔒 API Authentication Middleware**: High-performance API authentication validating client IDs and encrypted API keys stored in MongoDB with low-latency Redis caching.

---

## 🛠️ Technology Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python >= 3.13)
- **Dependency Manager**: [Astral `uv`](https://docs.astral.sh/uv/)
- **Orchestration / Agents**: [Pydantic AI](https://ai.pydantic.dev/)
- **Database**: [MongoDB](https://www.mongodb.com/) (using `pymongo`)
- **Caching & Rate Limiting**: [Redis](https://redis.io/)
- **Task Queue**: [Hatchet SDK](https://docs.hatchet.run/)
- **Observability**: [Logfire](https://logfire.dev/)
- **Graph Database**: [Neo4j](https://neo4j.com/)

---

## 📖 Project Documentation

To make getting started and maintaining Kita API as smooth as possible, the documentation is divided into the following guides:

### 💻 [Local Development Guide](./DEVELOPMENT.md)
Contains step-by-step setup guides for running the API locally, managing environment variables, utilizing `uv` for dependencies, and **generating or calling credentials for API clients**.

### 🚀 [Production Deployment Guide](./DEPLOYMENT.md)
Covers production deployment options, including Docker configurations, PaaS (Heroku/Render/Dokku) workflows, configuration check-lists, and exporting dependencies for production.

---

## 🚦 Quick Start

For full setup steps, environment variables, and dependencies, refer to [DEVELOPMENT.md](./DEVELOPMENT.md).

1. **Setup Environment**
   ```bash
   cp .env.example .env.local
   # Update variables in .env.local
   ```

2. **Sync Virtual Environment**
   ```bash
   uv sync
   ```

3. **Run Application**
   ```bash
   uv run uvicorn main:app --reload
   ```

---

## 🗺️ Repository Structure

* `app/` — Application source code.
  * [middleware/](./app/middleware) — Request interceptors, including client-key validation.
  * [routes/](./app/routes) — API routers (chats, files, events, memory, etc.).
  * [services/](./app/services) — Core business services (caching, encryption, etc.).
  * [db.py](./app/db.py) — MongoDB connection layer & Tenant wrapper.
* `scripts/` — Utility automation scripts.
  * [generate_client.py](./scripts/generate_client.py) — CLI to issue credentials to API clients.
* [main.py](./main.py) — Application entrypoint and middleware definition.
* [pyproject.toml](./pyproject.toml) — Project dependencies managed by `uv`.
