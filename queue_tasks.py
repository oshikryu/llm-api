"""arq task functions for async job processing."""

import httpx

from config import MODEL_NAME, REQUEST_TIMEOUT, STREAM_TIMEOUT


async def _record_worker_usage(ctx: dict, user_id: str, prompt_tokens: int, completion_tokens: int):
    """Record token usage from worker context."""
    redis = ctx.get("redis")
    if redis is None or not user_id:
        return
    key = f"llmapi:usage:{user_id}:{MODEL_NAME}"
    if prompt_tokens:
        await redis.hincrby(key, "prompt_tokens", prompt_tokens)
    if completion_tokens:
        await redis.hincrby(key, "completion_tokens", completion_tokens)


async def process_chat_completion(ctx: dict, request_data: dict):
    """Worker task: execute a chat completion against llama-server."""
    client: httpx.AsyncClient = ctx["http_client"]
    user_id = request_data.pop("_user_id", "")
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in request_data["messages"]
    ]
    full_content = ""
    total_completion_tokens = 0
    total_prompt_tokens = 0
    max_continuations = 25
    continue_until_done = request_data.get("continue_until_done", False)
    timeout = STREAM_TIMEOUT if continue_until_done else REQUEST_TIMEOUT

    for iteration in range(max_continuations):
        payload = {
            "messages": messages,
            "max_tokens": request_data.get("max_tokens", 512),
            "temperature": request_data.get("temperature", 0.7),
            "top_p": request_data.get("top_p", 0.9),
        }
        stop = request_data.get("stop")
        if stop:
            payload["stop"] = stop

        response = await client.post(
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

        if not continue_until_done:
            break
        if finish_reason == "stop":
            break
        if len(content.strip()) < 10:
            break
        messages.append({"role": "assistant", "content": content})

    await _record_worker_usage(ctx, user_id, total_prompt_tokens, total_completion_tokens)

    return {
        "message": {"role": "assistant", "content": full_content},
        "tokens_generated": total_completion_tokens,
        "tokens_prompt": total_prompt_tokens,
    }


async def process_completion(ctx: dict, request_data: dict):
    """Worker task: execute a text completion against llama-server."""
    client: httpx.AsyncClient = ctx["http_client"]
    user_id = request_data.pop("_user_id", "")
    payload = {
        "prompt": request_data["prompt"],
        "n_predict": request_data.get("max_tokens", 512),
        "temperature": request_data.get("temperature", 0.7),
        "top_p": request_data.get("top_p", 0.9),
    }
    stop = request_data.get("stop")
    if stop:
        payload["stop"] = stop

    response = await client.post("/completion", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    prompt_tokens = data.get("tokens_evaluated", 0)
    completion_tokens = data.get("tokens_predicted", 0)
    await _record_worker_usage(ctx, user_id, prompt_tokens, completion_tokens)

    return {
        "text": data.get("content", ""),
        "tokens_generated": completion_tokens,
        "tokens_prompt": prompt_tokens,
    }
