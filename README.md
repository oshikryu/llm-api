# Local LLM API

API wrapper for querying a local llama-server instance.

## Setup

1. Start the llama-server:
```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --port 8080 --threads 10 --ctx-size 4096 --n-predict 512
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
    "temperature": 0.7
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
python client.py --chat "Hello, how are you?"
python client.py --health
```

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API docs.
