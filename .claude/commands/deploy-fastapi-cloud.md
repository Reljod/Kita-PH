---
description: Deploy the Kita API to FastAPI Cloud, or diagnose a deployment that will not serve.
argument-hint: "[app id, defaults to FASTAPI_CLOUD_APP_ID]"
---

Deploy or diagnose using the **deploy-fastapi-cloud** skill at
`.agents/skills/deploy-fastapi-cloud/SKILL.md`. Read that skill and follow it.

App ID: $ARGUMENTS (if empty, use `FASTAPI_CLOUD_APP_ID` from the environment).

Before deploying, confirm `FASTAPI_CLOUD_TOKEN` is set in the environment and
never written into the repo. After deploying, verify with `/docs` and the
status-code table in the skill rather than assuming the outcome.
