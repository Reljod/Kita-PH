# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

# Testing

Run all tests: `uv run python -m unittest discover tests/ -v`

Run a single test file: `uv run python -m unittest tests.test_<name> -v`

## Test conventions
- Use `unittest.TestCase` + FastAPI `TestClient(raise_server_exceptions=False)`
- Create a fresh `FastAPI()` app in `setUpClass`, import and include the actual route router
- Override all service dependencies via `app.dependency_overrides` with `MagicMock`/`AsyncMock`
- Use `setup_error_handlers(app)` for consistent error response formatting
- SSE endpoints: mock the service method as an async generator, assert `text/event-stream` content-type

## Known issues (fixed)
- Routes with `try/except Exception` blocks incorrectly catch `HTTPException` raised for 4xx responses.
  Fix: add `except HTTPException: raise` before the generic `except Exception` handler.
- `OrgCreate` model uses `org_name`, not `name` (was `org_in.name` at line 64, fixed).
