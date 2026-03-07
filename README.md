# Local LLM API

API wrapper for querying a local llama-server instance, with connection pooling, concurrency control, async job queuing, and Docker-based multi-instance scaling.

## Setup

### Local Development

1. Start the llama-server:
```bash
llama-server \
  -m gpt-oss-20b-mxfp4.gguf \
  --port 8080 \
  -ngl 99 \
  --ctx-size 8192 \
  --batch-size 512 \
  --ubatch-size 128
```

When running moondream:
```bash
 llama-server \
  -m moondream2-text-model-f16.gguf \
  --mmproj moondream2-mmproj-f16.gguf \
  --port 8080 \
  -ngl 99 \
  --ctx-size 2048 \
  --batch-size 512 \
  --ubatch-size 128
```

2. Install Python dependencies and start the API:
```bash
source venv/bin/activate
pip install -r requirements.txt

# Development (single worker, auto-reload)
uvicorn api:app --reload --port 8000

# Production / load testing (multiple workers)
uvicorn api:app --port 8000 --workers 4
```

### Docker (Multi-Instance)

Launch the full stack with two llama-server instances, nginx load balancer, Redis, the API, and a queue worker. Models are loaded from `~/projects/models/`.

**Moondream2 (default):**
```bash
docker-compose up --build
```

**Llama 3.2 1B Instruct:**
```bash
docker-compose --env-file .env.llama up --build
```

**Switch back to Moondream2:**
```bash
docker-compose --env-file .env.moondream up --build
```

The default `.env` file loads Moondream2. Use `--env-file` to select a different model configuration.

This starts:
- **llama-server-1 / llama-server-2** — two llama.cpp inference backends (built from source)
- **nginx** — `least_conn` load balancer fronting both servers on port 8080
- **redis** — backing store for the async job queue
- **api** — FastAPI application on port 8000 (4 gunicorn/uvicorn workers)
- **worker** — arq queue worker for async job processing

Available model env files:
| File | Model | Context |
|---|---|---|
| `.env` (default) | Moondream2 (multimodal) | 2048 |
| `.env.llama` | Llama 3.2 1B Instruct Q8 | 8192 |
| `.env.moondream` | Moondream2 (multimodal) | 2048 |

## Authentication & User Management

The API supports multi-user API key authentication with per-user rate limiting and token usage tracking. All auth state is stored in Redis.

### Bootstrap Users

Seed initial users (admin + regular users with rate/token limits):

```bash
python scripts/seed_admin.py
```

This creates four users with API keys printed to stdout (save them — shown only once):

| User | Role | Rate Limit (min/hr/day) | Token Limit |
|------|------|------------------------|-------------|
| Admin | admin | default (60/min) | unlimited |
| Alice | user | 60/min, 500/hr, 5000/day | 500,000 |
| Bob | user | 30/min, 200/hr, 2000/day | 100,000 |
| Carol (Power User) | user | 120/min, 2000/hr | unlimited |

Edit the `SEED_USERS` list in `scripts/seed_admin.py` to customize.

### Using API Keys

Pass your key via `Authorization: Bearer <key>` or `X-API-Key: <key>`:

```bash
curl -H "Authorization: Bearer llm-YOUR_KEY" http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### Admin Endpoints

All admin endpoints require an admin API key:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/admin/users` | POST | Create user |
| `/admin/users` | GET | List all users |
| `/admin/users/{user_id}` | GET | User details |
| `/admin/users/{user_id}` | DELETE | Delete user + all keys |
| `/admin/users/{user_id}/keys` | POST | Create API key (raw key shown once) |
| `/admin/users/{user_id}/keys/{key_prefix}` | DELETE | Revoke a key |
| `/admin/users/{user_id}/limits/{model}` | PUT | Set token limit (`{"max_total_tokens": N}`) |
| `/admin/users/{user_id}/rate-limit` | PUT | Set rate limits (`{"requests_per_minute": N, "requests_per_hour": N, "requests_per_day": N}`) |
| `/admin/users/{user_id}/usage` | GET | Token usage per model |
| `/admin/users/{user_id}/history` | GET | API call history (paginated) |

### Billing & Usage (Self-Service)

Any authenticated user can view their own token consumption and limits:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/usage` | GET | Token usage across all models |
| `/usage/{model}` | GET | Token usage for a specific model |
| `/limits` | GET | Token limits with remaining budget and usage percentage |
| `/billing` | GET | Full summary: usage, limits, and current rate limit status |
| `/history` | GET | Recent API call history with per-request token usage (paginated) |

```bash
# View all usage
curl -H "Authorization: Bearer llm-YOUR_KEY" http://localhost:8000/usage

# View billing summary
curl -H "Authorization: Bearer llm-YOUR_KEY" http://localhost:8000/billing
```

Response from `/billing`:
```json
{
    "user_id": "abc123",
    "name": "Alice",
    "usage": [
        {
            "model": "default",
            "prompt_tokens": 5000,
            "completion_tokens": 3000,
            "total_tokens": 8000
        }
    ],
    "limits": [
        {
            "model": "default",
            "max_total_tokens": 500000,
            "current_total_tokens": 8000,
            "remaining_tokens": 492000,
            "usage_percent": 1.6
        }
    ],
    "rate_limit": {
        "requests_per_minute": 60,
        "requests_per_hour": 500,
        "requests_per_day": 5000,
        "current_requests_per_minute": 5,
        "current_requests_per_hour": 42,
        "current_requests_per_day": 310
    }
}
```

Models with no token limit set show `remaining_tokens: null` and `usage_percent: null`.

Response from `/history?offset=0&limit=50`:
```json
{
    "user_id": "abc123",
    "total": 142,
    "offset": 0,
    "limit": 50,
    "history": [
        {
            "timestamp": 1700000060.0,
            "endpoint": "/chat",
            "model": "default",
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
            "status": "ok"
        }
    ]
}
```

History is capped at the most recent 1000 entries per user. Use `offset` and `limit` (max 100) for pagination.

### Disabling Auth

Set `AUTH_ENABLED=false` to disable authentication (all requests treated as admin):

```bash
AUTH_ENABLED=false uvicorn api:app --reload --port 8000
```

## Load Testing

Simulate hundreds of requests per second across multiple users to validate rate limiting, token limits, and concurrency handling.

### Quick Start

```bash
# Requires Redis running and API server up
python scripts/load_test.py                            # gate mode (default): 200 rps, 5s
python scripts/load_test.py --rps 1000 --duration 10   # 1000 rps gate stress test
python scripts/load_test.py --mode llm --rps 20        # llm mode: real inference, lower rps
```

### Modes

| Mode | What it tests | Endpoint | Typical RPS | Default timeout |
|------|--------------|----------|-------------|-----------------|
| `gate` (default) | Full auth + rate limit + token limit pipeline, **zero LLM inference**. Pure request-handling throughput. | `GET /me` | 200–1000+ | 5s |
| `auth` | Auth pipeline + minimal LLM work (`max_tokens=4`). Still bottlenecked by llama-server. | `/chat`, `/completion`, `/query` | 20–100 | 10s |
| `llm` | Full LLM inference (`max_tokens=16`). Bottlenecked by llama-server throughput (~4 parallel slots). | `/chat`, `/completion`, `/query` | 10–50 | 120s |

Use `gate` mode for high-RPS stress testing of auth and rate limiting without LLM saturation. Use `auth` to test the full request path with minimal inference. Use `llm` to benchmark end-to-end throughput.

### Options

```bash
python scripts/load_test.py --rps 1000 --duration 10                  # 1000 rps gate stress
python scripts/load_test.py --mode auth --rps 50                      # auth + minimal LLM
python scripts/load_test.py --mode llm --rps 20 --duration 10         # LLM throughput test
python scripts/load_test.py --base-url http://api-host:8000 --rps 300 # remote target
python scripts/load_test.py --mode llm --timeout 180                  # custom timeout
```

### How It Works

1. Seeds **fresh test users** into Redis (Admin, Alice, Bob, Carol) with realistic rate/token limits
2. Distributes traffic with **weighted round-robin** — Carol 40%, Alice 30%, Bob 20%, Admin 10%
3. Rotates across `/chat`, `/completion`, and `/query` endpoints
4. Paces requests to hit the target RPS with a concurrency cap

### Output

Progress goes to stderr; structured JSON results go to stdout (pipe-friendly):

```bash
python scripts/load_test.py --rps 200 --duration 5 | jq .
```

Output structure:

```json
{
  "summary": {
    "mode": "gate",
    "target_rps": 200,
    "duration_seconds": 5,
    "total_requests": 1000,
    "wall_time_seconds": 5.12,
    "actual_rps": 195.3,
    "total_success": 620,
    "total_rate_limited_429": 340,
    "total_token_limited_403": 15,
    "total_timeout": 0,
    "total_errors": 25,
    "latency_ms": { "min": 1.2, "median": 8.5, "p95": 45.3, "p99": 120.1, "max": 350.0 }
  },
  "users": [
    {
      "user_id": "abc123...",
      "name": "Alice",
      "total_requests": 300,
      "success": 58,
      "rate_limited_429": 240,
      "token_limited_403": 2,
      "auth_error_401": 0,
      "server_error_5xx": 0,
      "timeout": 0,
      "latency_ms": { "min": 1.5, "median": 7.2, "p95": 30.0, "p99": 85.0, "max": 120.0 }
    }
  ]
}
```

Each entry in `users` includes full per-user breakdowns: success count, 429s (rate limited), 403s (token limit exceeded), errors, timeouts, and latency percentiles.

See [`diagrams/load_test.mmd`](diagrams/load_test.mmd) for a visual diagram of the load test architecture.

## Configuration

All settings are controlled via environment variables (see `config.py`):

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | Base URL of llama-server (or nginx LB) |
| `LLAMA_PARALLEL_SLOTS` | `4` | Max concurrent LLM requests (semaphore limit) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection for async queue + auth state |
| `REQUEST_TIMEOUT` | `120` | Default request timeout (seconds) |
| `STREAM_TIMEOUT` | `600` | Timeout for streaming requests (seconds) |
| `AUTH_ENABLED` | `true` | Enable/disable API key authentication |
| `MODEL_NAME` | `default` | Model name for per-model token tracking |
| `DEFAULT_RATE_LIMIT` | `60` | Default requests/minute when no per-user config |
| `DEFAULT_RATE_LIMIT_HOUR` | `0` | Default requests/hour (0 = unlimited) |
| `DEFAULT_RATE_LIMIT_DAY` | `0` | Default requests/day (0 = unlimited) |
| `DEFAULT_TOKEN_LIMIT` | `0` | Default token limit (0 = unlimited) |

## Architecture

### Connection Pooling
A single shared `httpx.AsyncClient` is created at startup with configurable connection pool limits, eliminating the overhead of creating a new HTTP client per request.

### Concurrency Control
An `asyncio.Semaphore` (sized to `LLAMA_PARALLEL_SLOTS`) gates all LLM requests. This prevents overloading the llama-server beyond its slot capacity. Streaming requests hold the semaphore for their full duration.

### Async Job Queue
For high-throughput scenarios, requests can be submitted to a Redis-backed queue (via arq) and polled for results. This decouples request acceptance from processing and enables horizontal scaling of workers.

## Endpoints

### Health Check
```
GET /health
```

### Text Completion
```
POST /completion
{
    "prompt": "Your prompt here",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["\\n\\n"]
}
```

### Chat Completion (OpenAI-compatible)
```
POST /chat
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "continue_until_done": true
}
```

Set `continue_until_done: true` to keep generating until the model finishes naturally, instead of stopping at `max_tokens`.

### Streaming Chat
```
POST /chat/stream
```
Same request body as `/chat`. Returns a `text/plain` stream of generated tokens.

### Async Job Submission

Submit a job to the Redis queue and poll for results:

```
POST /chat/async
{
    "messages": [{"role": "user", "content": "Hello"}]
}
```

Response:
```json
{"job_id": "abc-123", "status": "queued"}
```

```
POST /completion/async
{
    "prompt": "Your prompt here"
}
```

Poll status:
```
GET /jobs/{job_id}
```

Response (when complete):
```json
{
    "job_id": "abc-123",
    "status": "complete",
    "result": {
        "message": {"role": "assistant", "content": "..."},
        "tokens_generated": 42,
        "tokens_prompt": 10
    }
}
```

### RAG (Retrieval-Augmented Generation)

First, ingest documents into a collection:
```
POST /documents
{
    "documents": [
        {
            "content": "The O-1A visa is for individuals with extraordinary ability in sciences, education, business, or athletics. Applicants must demonstrate sustained national or international acclaim.",
            "metadata": {"source": "immigration-guide", "topic": "o1a"}
        },
        {
            "content": "O-1A criteria include: awards, membership in associations, published material, judging, original contributions, scholarly articles, employment in a critical role, and high salary.",
            "metadata": {"source": "immigration-guide", "topic": "o1a"}
        }
    ],
    "collection": "visa-docs",
    "chunk_size": 500
}
```

Response:
```json
{
    "documents_processed": 2,
    "chunks_created": 2,
    "chunk_ids": [
        "d1f2e3a4_chunk_0",
        "b5c6d7e8_chunk_0"
    ],
    "collection": "visa-docs"
}
```

Then query with RAG to get an LLM answer grounded in your documents:
```
POST /rag
{
    "query": "What are the requirements for an O-1A visa?",
    "collection": "visa-docs",
    "n_results": 3,
    "include_sources": true
}
```

Response:
```json
{
    "query": "What are the requirements for an O-1A visa?",
    "answer": "Based on the provided context, the O-1A visa is for individuals with extraordinary ability in sciences, education, business, or athletics. Applicants must demonstrate sustained national or international acclaim. The criteria include: awards, membership in associations, published material, judging, original contributions, scholarly articles, employment in a critical role, and high salary.",
    "sources": [
        {
            "chunk_id": "d1f2e3a4_chunk_0",
            "content": "The O-1A visa is for individuals with extraordinary ability...",
            "metadata": {"source": "immigration-guide", "topic": "o1a"},
            "distance": 0.25
        },
        {
            "chunk_id": "b5c6d7e8_chunk_0",
            "content": "O-1A criteria include: awards, membership in associations...",
            "metadata": {"source": "immigration-guide", "topic": "o1a"},
            "distance": 0.31
        }
    ],
    "tokens_generated": 87,
    "tokens_prompt": 312
}
```

You can also search without LLM generation:
```
POST /search
{
    "query": "visa criteria",
    "collection": "visa-docs",
    "n_results": 5
}
```

### Analyze (convenience endpoint)
```
POST /analyze?prompt=Summarize this&context=Your text here
```

## Client Usage

```python
from client import completion, chat, health_check

# Check health
print(health_check())

# Text completion
result = completion("What is machine learning?")
print(result)

# Chat
result = chat([
    {"role": "user", "content": "Explain O-1A visa criteria"}
])
print(result)
```

CLI usage:
```bash
python client.py "What is machine learning?"
python client.py --chat "Hello, how are you?"  # Continues until done by default
python client.py --chat --no-continue "Quick question"  # Stop at max_tokens
python client.py --max-tokens 1024 "Your prompt"
python client.py --health
```

## Testing

```bash
pytest tests/ -v
```

Tests mock the shared `http_client` and `llm_semaphore` — no running llama-server or Redis required.

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API docs.
