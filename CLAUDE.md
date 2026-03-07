# CLAUDE.md

## Project Overview

FastAPI wrapper around a local llama-server (llama.cpp) with RAG support, async job queuing, and Docker infrastructure for multi-instance scaling.

## Key Commands

- **Run API**: `uvicorn api:app --reload --port 8000`
- **Run tests**: `pytest tests/ -v`
- **Run queue worker**: `arq queue_worker.WorkerSettings`
- **Docker full stack**: `docker-compose up --build`
- **Activate venv**: `source venv/bin/activate`

## Project Structure

```
api.py              # Main FastAPI app — endpoints, lifespan, shared httpx client + semaphore
auth.py             # Auth module — API key validation, rate limiting, token limits
admin.py            # Admin endpoints — user/key/limit CRUD
config.py           # Centralized env-var configuration
queue_tasks.py      # arq worker task functions (process_chat_completion, process_completion)
queue_worker.py     # arq WorkerSettings — run as separate process
client.py           # CLI/Python client for the API
gunicorn.conf.py    # Gunicorn config (4 uvicorn workers)
Dockerfile          # Python 3.13-slim image
docker-compose.yml  # Full stack: 2x llama-server, nginx LB, redis, api, worker
nginx/nginx.conf    # least_conn load balancer for llama-server instances
rag/                # RAG module
  __init__.py       # Exports rag_router
  router.py         # /documents, /search, /rag, /collections endpoints
  models.py         # Pydantic models for RAG
  chunking.py       # Text chunking logic
  vectorstore.py    # ChromaDB wrapper
scripts/
  seed_admin.py     # Bootstrap CLI — create initial users and API keys
  generate_usage.py # Generate token usage via real API requests
  load_test.py      # Load test — multi-user, multi-mode, structured output
diagrams/
  architecture.mmd  # System architecture diagram (Mermaid)
  load_test.mmd     # Load test flow diagram (Mermaid)
tests/
  conftest.py       # Fixtures: mock_httpx (patches api.http_client), client, response helpers
  test_api.py       # Tests for core API endpoints
  test_auth.py      # Tests for auth module
  test_admin.py     # Tests for admin endpoints
  test_billing.py   # Tests for billing/usage endpoints
  test_rag.py       # Tests for RAG endpoints
  test_concurrency.py # Tests for semaphore concurrency control
  test_queue.py     # Tests for async job endpoints
```

## Architecture Decisions

- **Shared httpx.AsyncClient**: Single pooled client created in `lifespan`, stored as `api.http_client`. All endpoints use it via `_llm_post()` helper.
- **Semaphore concurrency**: `api.llm_semaphore` (sized to `LLAMA_PARALLEL_SLOTS`) gates all LLM calls. Streaming holds it for full duration.
- **arq for async jobs**: Redis-backed queue with `/chat/async`, `/completion/async`, and `/jobs/{job_id}` endpoints. Gracefully degrades if Redis is unavailable (`arq_pool = None`).
- **Config via env vars**: All in `config.py` — `LLAMA_SERVER_URL`, `LLAMA_PARALLEL_SLOTS`, `REDIS_URL`, `REQUEST_TIMEOUT`, `STREAM_TIMEOUT`.

## Testing Patterns

- Tests mock `api.http_client` (an `AsyncMock`) and `api.llm_semaphore` (a real `asyncio.Semaphore(4)`) via `patch()`.
- No `__aenter__`/`__aexit__` mocking needed — the shared client is used directly, not as a context manager.
- RAG tests use the `mock_httpx` fixture from conftest for LLM calls plus `@patch("rag.router.vectorstore")` for ChromaDB.
- Queue tests patch `api.arq_pool` with an `AsyncMock`.
- Helper functions `make_completion_response()` and `make_chat_response()` in `conftest.py` build mock responses.

## Style / Conventions

- Python 3.13, uses `X | None` union syntax
- Pydantic v2 models
- pytest + pytest-asyncio for async tests
- No type stubs for third-party packages (pyright reportMissingImports warnings from venv are expected)
