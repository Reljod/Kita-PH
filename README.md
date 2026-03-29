# Kita API

Modern FastAPI project for LLM-powered services, now managed with `uv`.

## Running the Application

To run the FastAPI server, use `uv run`:

```bash
uv run uvicorn main:app --reload
```

## Dependency Management

This project uses `uv` for lightning-fast dependency management.

### Adding New Packages
```bash
uv add <package_name>
```

### Syncing the Environment
```bash
uv sync
```

### Running Scripts
```bash
uv run python <script_name>.py
```
