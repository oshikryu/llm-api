import os

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_PARALLEL_SLOTS = int(os.getenv("LLAMA_PARALLEL_SLOTS", "4"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
STREAM_TIMEOUT = float(os.getenv("STREAM_TIMEOUT", "600"))
