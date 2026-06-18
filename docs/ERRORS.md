# Kita API - Enterprise Error Codes Reference

This document catalogs the custom enterprise error codes returned by the Kita API. The Kita API wraps all error responses in a standardized JSON payload structure. Standard HTTP status codes (4XX, 5XX) are still returned at the protocol level, but client applications should inspect the internal `code` string for robust, domain-specific error handling.

## Standard Error Response Format

All API errors return the following standard JSON format:

```json
{
  "error": {
    "code": "AGENT_RUN_FAILED",
    "message": "Detailed description of the error.",
    "details": {
      "extra_field": "specific error context"
    },
    "trace_id": "4020a560-6b6c-485a-8b0b-da54594a1370"
  }
}
```

- **`code`**: A unique, uppercase string identifying the specific error category and reason.
- **`message`**: A developer/user-friendly error message.
- **`details`**: An optional dictionary containing structured debugging information (e.g., input validation failures).
- **`trace_id`**: The request correlation ID (UUID), which can be used to track the request in server logs (Logfire, stdout, etc.).

---

## Error Codes Catalog

### 1. System & Operational Errors (`SYSTEM_`)

These errors indicate operational failures, database communication issues, validation bugs, or environmental misconfigurations.

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `SYSTEM_INTERNAL_ERROR` | 500 Internal Server Error | Generic, unhandled exceptions occurred in the server. |
| `SYSTEM_DATABASE_ERROR` | 500 Internal Server Error | The API failed to perform an operation on MongoDB. |
| `SYSTEM_REDIS_ERROR` | 500 Internal Server Error | Cache retrieval or connection with Redis failed. |
| `SYSTEM_VALIDATION_ERROR` | 422 Unprocessable Entity | Input request validation failed (e.g., Pydantic parsing errors). |
| `SYSTEM_CONFIG_ERROR` | 500 Internal Server Error | A required server environment variable is missing or invalid. |

### 2. Authentication & Authorization Errors (`AUTH_`)

These errors cover authentication checks, access privileges, and API key validations.

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `AUTH_UNAUTHORIZED` | 401 Unauthorized | Missing or invalid authentication token. |
| `AUTH_FORBIDDEN` | 403 Forbidden | Authenticated user/client does not have permission to access the resource. |
| `AUTH_INVALID_KEY_OR_ID` | 401 Unauthorized | Invalid `x-client-id` or `x-api-key` headers. |

### 3. Agent Management & Runtime Errors (`AGENT_`)

These errors relate to the agent definitions, versioning, and execution engine runs.

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `AGENT_NOT_FOUND` | 404 Not Found | The requested agent ID could not be found. |
| `AGENT_VERSION_NOT_FOUND` | 404 Not Found | The requested version for the agent does not exist. |
| `AGENT_RUN_FAILED` | 500 Internal Server Error | Non-streaming agent run failed to complete execution. |
| `AGENT_RUN_STREAM_FAILED` | 500 Internal Server Error | Streaming agent run failed midway through token streaming. |

### 4. Tool Management & Execution Errors (`TOOL_`)

These errors represent tool registration issues, missing tools, or failures during tool execution.

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `TOOL_NOT_FOUND` | 404 Not Found | The tool is not registered or cannot be resolved. |
| `TOOL_REGISTRATION_FAILED`| 400 Bad Request | Tool registration inputs are invalid or duplicate registration attempted. |
| `TOOL_EXECUTION_FAILED` | 500 Internal Server Error | Generic error indicating a tool execution failed. |
| `TOOL_AGENT_CREATION_FAILED`| 500 Internal Server Error | The agent creation tool encountered an exception. |
| `TOOL_DELEGATION_FAILED` | 500 Internal Server Error | Task delegation tool failed to run or communicate with the target agent. |
| `TOOL_FILE_OPERATION_FAILED`| 500 Internal Server Error | File system, read, or write tool operations failed. |
| `TOOL_GRAPH_RAG_QUERY_FAILED`| 500 Internal Server Error | Graph RAG search tool failed to query nodes or edges. |
| `TOOL_LLM_COMPLETION_FAILED`| 500 Internal Server Error | Inner LLM call within a tool failed or timed out. |
| `TOOL_MEMORY_OPERATION_FAILED`| 500 Internal Server Error | Vector memory write, read, or search tool execution failed. |
| `TOOL_PARSE_FAILED` | 500 Internal Server Error | LlamaParse or document parsing tool execution failed. |
| `TOOL_WEB_SEARCH_FAILED` | 500 Internal Server Error | Web search tool (Serper API query) failed. |

### 5. RAG & Retrieval Errors (`RAG_`)

These errors concern vector databases, enrichments, and vector generation pipelines.

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `RAG_QUERY_FAILED` | 500 Internal Server Error | Querying vector databases or reranking search results failed. |
| `RAG_ENRICHMENT_FAILED` | 500 Internal Server Error | Parsing enrichment results or updating database nodes failed. |

### 6. Memory Errors (`MEMORY_`)

These errors cover agent episodic or long-term storage records.

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `MEMORY_NOT_FOUND` | 404 Not Found | The requested memory record (RAG document) does not exist. |
| `MEMORY_OPERATION_FAILED` | 500 Internal Server Error | Saving, updating, or deleting a memory record failed. |

### 7. File Errors (`FILE_`)

These errors relate to storage management, upload sessions, and parsing operations.

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `FILE_NOT_FOUND` | 404 Not Found | The requested file does not exist. |
| `FILE_UPLOAD_FAILED` | 500 Internal Server Error | Uploading file binary to storage (e.g., Supabase) failed. |
| `FILE_PARSING_FAILED` | 422 Unprocessable Entity | The file format was invalid, or parsing the document failed. |
