"""
API for querying local LLM running on llama-server.

Start the llama-server first:
    llama-server -hf ggml-org/gpt-oss-20b-GGUF --port 8080 --threads 10 --ctx-size 4096 --n-predict 512

Then run this API:
    uvicorn api:app --reload --port 8000
"""

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

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
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")


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


@app.post("/chat", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    """Generate a chat completion (OpenAI-compatible endpoint)."""
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }
    if request.stop:
        payload["stop"] = request.stop

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})

            return ChatResponse(
                message=ChatMessage(
                    role=message.get("role", "assistant"),
                    content=message.get("content", ""),
                ),
                tokens_generated=usage.get("completion_tokens", 0),
                tokens_prompt=usage.get("prompt_tokens", 0),
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(e)}")


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
