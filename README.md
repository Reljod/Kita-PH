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
## File Upload API

The API provides two methods for file uploads based on the file size.

### 1. Initiate Upload
Send a `POST` request to `/files/upload` with the file metadata:
```bash
curl -X POST "http://localhost:8000/files/upload" \
  -H "Authorization: Bearer <your_auth_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "document.pdf",
    "size": 1048576,
    "content_type": "application/pdf"
  }'
```

The response will provide an `upload_url`, `method`, and `token`.

### 2. Perform the Upload

#### **Standard Upload (Files < 6MB)**
The `method` will be `POST`. You must use **`PUT`** and **Raw Binary** data for the best consistency.
```bash
curl -X PUT "URL_FROM_API" \
  --data-binary "@/path/to/document.pdf"
```

#### **Resumable Upload (Files ≥ 6MB)**
The `method` will be `TUS`. You must use the **TUS protocol** (recommended: `tus-js-client`).

**Manual Example (via curl):**
```bash
# Step A: Initiate resumable session
curl -X POST "URL_FROM_API" \
  -H "Authorization: Bearer <token_from_api>" \
  -H "Tus-Resumable: 1.0.0" \
  -H "Upload-Length: <file_size>" \
  -H "Content-Type: application/offset+octet-stream" \
  -i

# Note the 'Location' header in the response, then:
# Step B: Upload data
curl -X PATCH "<location_header_url>" \
  -H "Tus-Resumable: 1.0.0" \
  -H "Upload-Offset: 0" \
  -H "Content-Type: application/offset+octet-stream" \
  --data-binary "@/path/to/large_file.mp4"
```
