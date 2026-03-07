"""
API for querying local LLM running on llama-server.

Start the llama-server first:
    llama-server -hf ggml-org/gpt-oss-20b-GGUF --port 8080 -ngl 99 --ctx-size 8192 --n-predict 512 --batch-size 512 --ubatch-size 128

Then run this API:
    uvicorn api:app --reload --port 8000
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Optional, AsyncGenerator

from config import (
    AUTH_ENABLED,
    LLAMA_SERVER_URL,
    LLAMA_PARALLEL_SLOTS,
    MODEL_NAME,
    REDIS_URL,
    REQUEST_TIMEOUT,
    STREAM_TIMEOUT,
)
from auth import (
    User,
    get_current_user,
    get_redis,
    check_rate_limit,
    check_token_limit,
    record_usage,
    get_rate_limit_config,
    get_rate_limit_counts,
)
from rag import rag_router

http_client: httpx.AsyncClient | None = None
llm_semaphore: asyncio.Semaphore | None = None
arq_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, llm_semaphore, arq_pool
    http_client = httpx.AsyncClient(
        base_url=LLAMA_SERVER_URL,
        limits=httpx.Limits(
            max_connections=LLAMA_PARALLEL_SLOTS * 2,
            max_keepalive_connections=LLAMA_PARALLEL_SLOTS,
        ),
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
    )
    llm_semaphore = asyncio.Semaphore(LLAMA_PARALLEL_SLOTS)
    try:
        app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        app.state.redis = None
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        arq_pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
    except Exception:
        arq_pool = None
    yield
    await http_client.aclose()
    if app.state.redis is not None:
        await app.state.redis.aclose()
    if arq_pool is not None:
        await arq_pool.aclose()


app = FastAPI(
    title="Local LLM API",
    description="API for querying local llama-server",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(rag_router, tags=["RAG"])

from admin import router as admin_router
app.include_router(admin_router, tags=["Admin"])


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the LLM")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    # str: plain text, e.g. "Hello"
    # list: OpenAI multimodal format, e.g.
    #   [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    content: Any = Field(..., description="Message content (string or multimodal content parts)")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")
    continue_until_done: bool = Field(default=False, description="Continue generating until model stops naturally")


class CompletionResponse(BaseModel):
    text: str
    tokens_generated: int
    tokens_prompt: int


class ChatResponse(BaseModel):
    message: ChatMessage
    tokens_generated: int
    tokens_prompt: int


async def _llm_post(path: str, *, json: dict, timeout: float = REQUEST_TIMEOUT):
    """POST to llama-server with semaphore-gated concurrency."""
    async with llm_semaphore:
        return await http_client.post(path, json=json, timeout=timeout)


async def _llm_stream(path: str, *, json: dict, timeout: float = STREAM_TIMEOUT):
    """Stream from llama-server, holding semaphore for the full duration."""
    return http_client.stream("POST", path, json=json, timeout=timeout)


@app.get("/health")
async def health_check():
    """Check if both this API and llama-server are running."""
    try:
        response = await http_client.get("/health", timeout=5.0)
        llama_status = response.json() if response.status_code == 200 else {"status": "error"}
    except Exception as e:
        llama_status = {"status": "unreachable", "error": str(e)}

    return {
        "api": "ok",
        "llama_server": llama_status,
    }


@app.get("/me")
async def get_me(
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Return current user info. Runs full auth + rate limit pipeline, no LLM."""
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)
    return {
        "user_id": user.user_id,
        "name": user.name,
        "is_admin": user.is_admin,
    }


# ---------------------------------------------------------------------------
# Billing & Usage (self-service)
# ---------------------------------------------------------------------------


class ModelUsage(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelLimit(BaseModel):
    model: str
    max_total_tokens: int
    current_total_tokens: int
    remaining_tokens: int | None = Field(
        description="Tokens remaining before limit, null if unlimited",
    )
    usage_percent: float | None = Field(
        description="Percentage of limit used, null if unlimited",
    )


class RateLimitStatus(BaseModel):
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    current_requests_per_minute: int
    current_requests_per_hour: int
    current_requests_per_day: int


class BillingResponse(BaseModel):
    user_id: str
    name: str
    usage: list[ModelUsage]
    limits: list[ModelLimit]
    rate_limit: RateLimitStatus


async def _scan_keys(redis, pattern: str) -> list[str]:
    """Collect all keys matching a pattern via SCAN."""
    keys: list[str] = []
    cursor = "0"
    while True:
        cursor, batch = await redis.scan(cursor=cursor, match=pattern, count=100)
        keys.extend(batch)
        if cursor == "0" or cursor == 0:
            break
    return keys


@app.get("/usage", response_model=list[ModelUsage])
async def get_my_usage(
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """View your own token usage across all models."""
    keys = await _scan_keys(redis, f"llmapi:usage:{user.user_id}:*")
    usages = []
    for key in keys:
        model = key.split(f"llmapi:usage:{user.user_id}:")[1]
        data = await redis.hgetall(key)
        prompt = int(data.get("prompt_tokens", 0))
        completion = int(data.get("completion_tokens", 0))
        usages.append(ModelUsage(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ))
    return usages


@app.get("/usage/{model}", response_model=ModelUsage)
async def get_my_usage_by_model(
    model: str,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """View your own token usage for a specific model."""
    data = await redis.hgetall(f"llmapi:usage:{user.user_id}:{model}")
    prompt = int(data.get("prompt_tokens", 0))
    completion = int(data.get("completion_tokens", 0))
    return ModelUsage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


@app.get("/limits", response_model=list[ModelLimit])
async def get_my_limits(
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """View your token limits and remaining budget across all models."""
    # Gather all models with either limits or usage
    limit_keys = await _scan_keys(redis, f"llmapi:limits:{user.user_id}:*")
    usage_keys = await _scan_keys(redis, f"llmapi:usage:{user.user_id}:*")

    models = set()
    for key in limit_keys:
        models.add(key.split(f"llmapi:limits:{user.user_id}:")[1])
    for key in usage_keys:
        models.add(key.split(f"llmapi:usage:{user.user_id}:")[1])

    results = []
    for model in sorted(models):
        limits = await redis.hgetall(f"llmapi:limits:{user.user_id}:{model}")
        max_total = int(limits.get("max_total_tokens", 0))

        usage = await redis.hgetall(f"llmapi:usage:{user.user_id}:{model}")
        current = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))

        remaining = None
        pct = None
        if max_total > 0:
            remaining = max(0, max_total - current)
            pct = round(current / max_total * 100, 1)

        results.append(ModelLimit(
            model=model,
            max_total_tokens=max_total,
            current_total_tokens=current,
            remaining_tokens=remaining,
            usage_percent=pct,
        ))
    return results


@app.get("/billing", response_model=BillingResponse)
async def get_my_billing(
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Full billing summary: usage, limits, and rate limit config."""
    # Usage
    usage_keys = await _scan_keys(redis, f"llmapi:usage:{user.user_id}:*")
    usages = []
    for key in usage_keys:
        model = key.split(f"llmapi:usage:{user.user_id}:")[1]
        data = await redis.hgetall(key)
        prompt = int(data.get("prompt_tokens", 0))
        completion = int(data.get("completion_tokens", 0))
        usages.append(ModelUsage(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ))

    # Limits
    limit_keys = await _scan_keys(redis, f"llmapi:limits:{user.user_id}:*")
    all_models = set()
    for key in limit_keys:
        all_models.add(key.split(f"llmapi:limits:{user.user_id}:")[1])
    for key in usage_keys:
        all_models.add(key.split(f"llmapi:usage:{user.user_id}:")[1])

    limits = []
    for model in sorted(all_models):
        lim_data = await redis.hgetall(f"llmapi:limits:{user.user_id}:{model}")
        max_total = int(lim_data.get("max_total_tokens", 0))
        usg_data = await redis.hgetall(f"llmapi:usage:{user.user_id}:{model}")
        current = int(usg_data.get("prompt_tokens", 0)) + int(usg_data.get("completion_tokens", 0))
        remaining = None
        pct = None
        if max_total > 0:
            remaining = max(0, max_total - current)
            pct = round(current / max_total * 100, 1)
        limits.append(ModelLimit(
            model=model,
            max_total_tokens=max_total,
            current_total_tokens=current,
            remaining_tokens=remaining,
            usage_percent=pct,
        ))

    # Rate limit
    rl_config = await get_rate_limit_config(user.user_id, redis)
    rl_counts = await get_rate_limit_counts(user.user_id, redis)

    return BillingResponse(
        user_id=user.user_id,
        name=user.name,
        usage=usages,
        limits=limits,
        rate_limit=RateLimitStatus(**rl_config, **rl_counts),
    )


@app.post("/completion", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Generate text completion from a prompt."""
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    result = await _do_completion(request)
    await record_usage(user.user_id, MODEL_NAME, result.tokens_prompt, result.tokens_generated, redis)
    return result


async def _do_completion(request: CompletionRequest) -> CompletionResponse:
    """Internal completion logic without auth dependencies."""
    payload = {
        "prompt": request.prompt,
        "n_predict": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }
    if request.stop:
        payload["stop"] = request.stop

    try:
        response = await _llm_post("/completion", json=payload)
        response.raise_for_status()
        data = response.json()

        return CompletionResponse(
            text=data.get("content", ""),
            tokens_generated=data.get("tokens_predicted", 0),
            tokens_prompt=data.get("tokens_evaluated", 0),
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(e)}")


def _looks_complete(content: str, tokens_generated: int, max_tokens: int) -> bool:
    """Heuristics to detect if a response looks complete."""
    # If we generated far fewer tokens than allowed, model likely finished
    if tokens_generated < max_tokens * 0.5:
        return True
    # Check for natural ending patterns
    stripped = content.rstrip()
    if stripped.endswith(('.', '!', '?', '```', '"', "'")):
        # Ends with sentence-ending punctuation or code block
        return True
    # Check for list/structured content that ends cleanly
    if stripped.endswith(':') or stripped.endswith('-'):
        return False  # Likely mid-list
    return False


@app.post("/chat", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Generate a chat completion (OpenAI-compatible endpoint)."""
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    result = await _do_chat(request)
    await record_usage(user.user_id, MODEL_NAME, result.tokens_prompt, result.tokens_generated, redis)
    return result


async def _do_chat(request: ChatRequest) -> ChatResponse:
    """Internal chat logic without auth dependencies."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    full_content = ""
    total_completion_tokens = 0
    total_prompt_tokens = 0
    max_continuations = 25  # Safety limit

    timeout = STREAM_TIMEOUT if request.continue_until_done else REQUEST_TIMEOUT
    try:
        for iteration in range(max_continuations):
            payload = {
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
            if request.stop:
                payload["stop"] = request.stop

            response = await _llm_post(
                "/v1/chat/completions", json=payload, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})
            finish_reason = choice.get("finish_reason", "stop")

            content = message.get("content", "")
            tokens_generated = usage.get("completion_tokens", 0)
            full_content += content
            total_completion_tokens += tokens_generated
            if iteration == 0:
                total_prompt_tokens = usage.get("prompt_tokens", 0)

            # Stop conditions
            if not request.continue_until_done:
                break
            if finish_reason == "stop":
                break
            if _looks_complete(content, tokens_generated, request.max_tokens):
                break
            # Empty or very short response means model is done
            if len(content.strip()) < 10:
                break

            # Append assistant response - model will continue from here
            messages.append({"role": "assistant", "content": content})

        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=full_content,
            ),
            tokens_generated=total_completion_tokens,
            tokens_prompt=total_prompt_tokens,
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(e)}")


@app.post("/chat/stream")
async def create_chat_completion_stream(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Stream chat completion with continuation support."""
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    async def generate() -> AsyncGenerator[bytes, None]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        max_continuations = 25
        total_tokens = 0

        try:
            for iteration in range(max_continuations):
                payload = {
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "stream": True,
                }
                if request.stop:
                    payload["stop"] = request.stop

                content_chunk = ""
                finish_reason = None
                tokens_generated = 0

                async with llm_semaphore:
                    async with http_client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json=payload,
                        timeout=STREAM_TIMEOUT,
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str.strip() == "[DONE]":
                                continue

                            try:
                                data = json.loads(data_str)
                                choice = data.get("choices", [{}])[0]
                                delta = choice.get("delta", {})
                                token = delta.get("content", "")
                                finish_reason = choice.get("finish_reason")

                                if token:
                                    content_chunk += token
                                    tokens_generated += 1
                                    yield token.encode("utf-8")
                            except json.JSONDecodeError:
                                continue

                total_tokens += tokens_generated

                # Stop conditions
                if not request.continue_until_done:
                    break
                if finish_reason == "stop":
                    break
                if _looks_complete(content_chunk, tokens_generated, request.max_tokens):
                    break
                if len(content_chunk.strip()) < 10:
                    break

                # Append for next iteration
                messages.append({"role": "assistant", "content": content_chunk})
        finally:
            await record_usage(user.user_id, MODEL_NAME, 0, total_tokens, redis)

    return StreamingResponse(generate(), media_type="text/plain")


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that provides clear, structured responses.

Guidelines:
- Be direct and concise
- Use markdown formatting for clarity
- For lists, use bullet points or numbered lists
- For technical content, use code blocks with language tags
- Structure long responses with headers (##)
- End with a brief summary if the response is lengthy"""


class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to send to the LLM")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt (overrides default)")
    output_format: Optional[str] = Field(default="markdown", description="Output format: markdown, json, or plain")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class QueryResponse(BaseModel):
    query: str
    response: str
    format: str
    tokens_generated: int
    structured: Optional[dict] = Field(default=None, description="Parsed JSON if output_format is json")


def _build_system_prompt(output_format: str, custom_prompt: Optional[str]) -> str:
    """Build system prompt based on output format."""
    if custom_prompt:
        return custom_prompt

    if output_format == "json":
        return """You are a helpful assistant that returns structured JSON responses.

Guidelines:
- Always return valid JSON
- Use consistent key names (snake_case)
- Include a "summary" field for the main answer
- Include a "details" field for additional information as an array
- Include a "confidence" field (high/medium/low) when applicable
- Do not include markdown formatting or code blocks, just raw JSON"""

    if output_format == "plain":
        return """You are a helpful assistant that provides clear, direct responses.

Guidelines:
- Be concise and direct
- No markdown formatting
- Use simple line breaks for structure
- Avoid unnecessary preamble"""

    return DEFAULT_SYSTEM_PROMPT


def _try_parse_json(content: str) -> Optional[dict]:
    """Try to parse JSON from response, handling code blocks."""
    content = content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """
    Query endpoint with structured output support.
    Automatically continues until the model finishes.
    """
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    output_format = request.output_format or "markdown"
    system_prompt = _build_system_prompt(output_format, request.system_prompt)

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=request.query),
    ]

    chat_request = ChatRequest(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        continue_until_done=True,
    )
    result = await _do_chat(chat_request)
    await record_usage(user.user_id, MODEL_NAME, result.tokens_prompt, result.tokens_generated, redis)

    response_content = result.message.content
    structured = None
    if output_format == "json":
        structured = _try_parse_json(response_content)

    return QueryResponse(
        query=request.query,
        response=response_content,
        format=output_format,
        tokens_generated=result.tokens_generated,
        structured=structured,
    )


@app.get("/query")
async def query_get(
    q: str,
    system: Optional[str] = None,
    format: str = "markdown",
    max_tokens: int = 512,
    temperature: float = 0.7,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """GET version for simple queries via URL."""
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    request = QueryRequest(
        query=q,
        system_prompt=system,
        output_format=format,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = await _do_query(request)
    return result


async def _do_query(request: QueryRequest) -> QueryResponse:
    """Internal query logic without auth dependencies."""
    output_format = request.output_format or "markdown"
    system_prompt = _build_system_prompt(output_format, request.system_prompt)

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=request.query),
    ]

    chat_request = ChatRequest(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        continue_until_done=True,
    )
    result = await _do_chat(chat_request)

    response_content = result.message.content
    structured = None
    if output_format == "json":
        structured = _try_parse_json(response_content)

    return QueryResponse(
        query=request.query,
        response=response_content,
        format=output_format,
        tokens_generated=result.tokens_generated,
        structured=structured,
    )


@app.post("/analyze")
async def analyze_text(
    prompt: str,
    context: Optional[str] = None,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """
    Convenience endpoint for analyzing text with optional context.
    Useful for O-1A visa criteria assessment tasks.
    """
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    full_prompt = ""
    if context:
        full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
    else:
        full_prompt = prompt

    request = CompletionRequest(prompt=full_prompt, max_tokens=512, temperature=0.3)
    result = await _do_completion(request)
    await record_usage(user.user_id, MODEL_NAME, result.tokens_prompt, result.tokens_generated, redis)
    return result


# ---------------------------------------------------------------------------
# Phase 2: Async job endpoints (Redis + arq)
# ---------------------------------------------------------------------------


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None


@app.post("/chat/async", response_model=AsyncJobResponse)
async def create_chat_completion_async(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Enqueue a chat completion job for async processing."""
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    if arq_pool is None:
        raise HTTPException(status_code=503, detail="Queue not available (Redis not connected)")
    job_id = str(uuid.uuid4())
    request_data = request.model_dump()
    request_data["_user_id"] = user.user_id
    await arq_pool.enqueue_job("process_chat_completion", request_data, _job_id=job_id)
    return AsyncJobResponse(job_id=job_id, status="queued")


@app.post("/completion/async", response_model=AsyncJobResponse)
async def create_completion_async(
    request: CompletionRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Enqueue a completion job for async processing."""
    await check_rate_limit(user.user_id, redis)
    await check_token_limit(user.user_id, MODEL_NAME, redis)

    if arq_pool is None:
        raise HTTPException(status_code=503, detail="Queue not available (Redis not connected)")
    job_id = str(uuid.uuid4())
    request_data = request.model_dump()
    request_data["_user_id"] = user.user_id
    await arq_pool.enqueue_job("process_completion", request_data, _job_id=job_id)
    return AsyncJobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user: User = Depends(get_current_user),
):
    """Poll the status of an async job."""
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="Queue not available (Redis not connected)")
    from arq.jobs import Job

    job = Job(job_id, arq_pool)
    info = await job.info()
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = info.status
    result = None
    if status == "complete":
        result = info.result
    return JobStatusResponse(job_id=job_id, status=status, result=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
