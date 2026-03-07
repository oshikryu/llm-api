import os

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_PARALLEL_SLOTS = int(os.getenv("LLAMA_PARALLEL_SLOTS", "4"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
STREAM_TIMEOUT = float(os.getenv("STREAM_TIMEOUT", "600"))
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
MODEL_NAME = os.getenv("MODEL_NAME", "default")
DEFAULT_RATE_LIMIT = int(os.getenv("DEFAULT_RATE_LIMIT", "60"))
DEFAULT_RATE_LIMIT_HOUR = int(os.getenv("DEFAULT_RATE_LIMIT_HOUR", "0"))   # 0 = unlimited
DEFAULT_RATE_LIMIT_DAY = int(os.getenv("DEFAULT_RATE_LIMIT_DAY", "0"))     # 0 = unlimited
DEFAULT_TOKEN_LIMIT = int(os.getenv("DEFAULT_TOKEN_LIMIT", "0"))
