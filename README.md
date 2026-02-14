# Local LLM API

API wrapper for querying a local llama-server instance.

## Setup

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

2. Install Python dependencies:
```bash
cd llm_api
pip install -r requirements.txt
```

3. Start the API:
```bash
uvicorn api:app --reload --port 8000
```

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

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API docs.
