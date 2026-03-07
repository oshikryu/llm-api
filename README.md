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

2. Install Python dependencies:
```bash
cd llm_api
pip install -r requirements.txt
```

3. Start the API:
```bash
uvicorn api:app --reload --port 8000
```

### Docker (Multi-Instance)

Launch the full stack with two llama-server instances, nginx load balancer, Redis, the API, and a queue worker:

```bash
docker-compose up --build
```

This starts:
- **llama-server-1 / llama-server-2** — two llama.cpp inference backends
- **nginx** — `least_conn` load balancer fronting both servers on port 8080
- **redis** — backing store for the async job queue
- **api** — FastAPI application on port 8000 (4 gunicorn/uvicorn workers)
- **worker** — arq queue worker for async job processing

Place your model file in the `models` Docker volume before starting.

## Configuration

All settings are controlled via environment variables (see `config.py`):

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | Base URL of llama-server (or nginx LB) |
| `LLAMA_PARALLEL_SLOTS` | `4` | Max concurrent LLM requests (semaphore limit) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection for async queue |
| `REQUEST_TIMEOUT` | `120` | Default request timeout (seconds) |
| `STREAM_TIMEOUT` | `600` | Timeout for streaming requests (seconds) |

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
