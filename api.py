"""
API for querying local LLM running on llama-server.

Start the llama-server first:
    llama-server -hf ggml-org/gpt-oss-20b-GGUF --port 8080 -ngl 99 --ctx-size 8192 --n-predict 512 --batch-size 512 --ubatch-size 128

Then run this API:
    uvicorn api:app --reload --port 8000
"""

import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator

app = FastAPI(
    title="Local LLM API",
    description="API for querying local llama-server",
    version="1.0.0",
)

LLAMA_SERVER_URL = "http://localhost:8080"


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the LLM")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


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


@app.get("/health")
async def health_check():
    """Check if both this API and llama-server are running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/health")
            llama_status = response.json() if response.status_code == 200 else {"status": "error"}
    except Exception as e:
        llama_status = {"status": "unreachable", "error": str(e)}

    return {
        "api": "ok",
        "llama_server": llama_status,
    }


@app.post("/completion", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Generate text completion from a prompt."""
    payload = {
        "prompt": request.prompt,
        "n_predict": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }
    if request.stop:
        payload["stop"] = request.stop

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LLAMA_SERVER_URL}/completion",
                json=payload,
            )
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
async def create_chat_completion(request: ChatRequest):
    """Generate a chat completion (OpenAI-compatible endpoint)."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    full_content = ""
    total_completion_tokens = 0
    total_prompt_tokens = 0
    max_continuations = 25  # Safety limit

    timeout = 600.0 if request.continue_until_done else 120.0
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for iteration in range(max_continuations):
                payload = {
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                }
                if request.stop:
                    payload["stop"] = request.stop

                response = await client.post(
                    f"{LLAMA_SERVER_URL}/v1/chat/completions",
                    json=payload,
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
async def create_chat_completion_stream(request: ChatRequest):
    """Stream chat completion with continuation support."""

    async def generate() -> AsyncGenerator[bytes, None]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        max_continuations = 25

        async with httpx.AsyncClient(timeout=600.0) as client:
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

                async with client.stream(
                    "POST",
                    f"{LLAMA_SERVER_URL}/v1/chat/completions",
                    json=payload,
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
async def query(request: QueryRequest):
    """
    Query endpoint with structured output support.
    Automatically continues until the model finishes.
    """
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
    result = await create_chat_completion(chat_request)

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
):
    """GET version for simple queries via URL."""
    request = QueryRequest(
        query=q,
        system_prompt=system,
        output_format=format,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return await query(request)


@app.post("/analyze")
async def analyze_text(prompt: str, context: Optional[str] = None):
    """
    Convenience endpoint for analyzing text with optional context.
    Useful for O-1A visa criteria assessment tasks.
    """
    full_prompt = ""
    if context:
        full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
    else:
        full_prompt = prompt

    request = CompletionRequest(prompt=full_prompt, max_tokens=512, temperature=0.3)
    return await create_completion(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
