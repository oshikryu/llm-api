"""Authentication, rate limiting, and token usage tracking."""

import hashlib
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from config import AUTH_ENABLED, DEFAULT_RATE_LIMIT, DEFAULT_RATE_LIMIT_HOUR, DEFAULT_RATE_LIMIT_DAY


@dataclass
class User:
    user_id: str
    name: str
    is_admin: bool


ANONYMOUS_USER = User(user_id="anonymous", name="Anonymous", is_admin=True)


async def get_redis(request: Request):
    return request.app.state.redis


async def get_current_user(
    request: Request,
    redis=Depends(get_redis),
) -> User:
    if not AUTH_ENABLED:
        return ANONYMOUS_USER

    api_key = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    if not api_key:
        api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    user_id = await redis.get(f"llmapi:apikey:{key_hash}")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_data = await redis.hgetall(f"llmapi:user:{user_id}")
    if not user_data:
        raise HTTPException(status_code=401, detail="User not found")

    return User(
        user_id=user_id,
        name=user_data.get("name", ""),
        is_admin=user_data.get("is_admin", "false") == "true",
    )


def _get_rate_tiers():
    return [
        ("requests_per_minute", 60_000, DEFAULT_RATE_LIMIT),
        ("requests_per_hour", 3_600_000, DEFAULT_RATE_LIMIT_HOUR),
        ("requests_per_day", 86_400_000, DEFAULT_RATE_LIMIT_DAY),
    ]


async def get_rate_limit_config(user_id: str, redis) -> dict:
    config = await redis.hgetall(f"llmapi:ratelimit_config:{user_id}")
    return {
        "requests_per_minute": int(config.get("requests_per_minute", DEFAULT_RATE_LIMIT)),
        "requests_per_hour": int(config.get("requests_per_hour", DEFAULT_RATE_LIMIT_HOUR)),
        "requests_per_day": int(config.get("requests_per_day", DEFAULT_RATE_LIMIT_DAY)),
    }


async def get_rate_limit_counts(user_id: str, redis) -> dict:
    key = f"llmapi:ratelimit:{user_id}"
    now_ms = int(time.time() * 1000)
    return {
        "current_requests_per_minute": await redis.zcount(key, now_ms - 60_000, "+inf"),
        "current_requests_per_hour": await redis.zcount(key, now_ms - 3_600_000, "+inf"),
        "current_requests_per_day": await redis.zcount(key, now_ms - 86_400_000, "+inf"),
    }


async def check_rate_limit(user_id: str, redis) -> None:
    key = f"llmapi:ratelimit:{user_id}"
    now_ms = int(time.time() * 1000)

    config = await redis.hgetall(f"llmapi:ratelimit_config:{user_id}")

    # Remove entries older than 24h (largest window)
    await redis.zremrangebyscore(key, 0, now_ms - 86_400_000)

    for tier_name, window_ms, default_limit in _get_rate_tiers():
        limit = int(config.get(tier_name, default_limit))
        if limit == 0:
            continue
        count = await redis.zcount(key, now_ms - window_ms, "+inf")
        if count >= limit:
            oldest = await redis.zrangebyscore(key, now_ms - window_ms, "+inf", start=0, num=1)
            retry_after = 0
            if oldest:
                oldest_ms = float(oldest[0])
                retry_after = max(0, int((oldest_ms + window_ms - now_ms) / 1000) + 1)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit_type": tier_name,
                    "limit": limit,
                    "current": count,
                    "retry_after_seconds": retry_after,
                },
            )

    await redis.zadd(key, {str(now_ms): now_ms})
    await redis.expire(key, 86460)


async def check_token_limit(user_id: str, model: str, redis) -> None:
    limits = await redis.hgetall(f"llmapi:limits:{user_id}:{model}")
    max_total = int(limits.get("max_total_tokens", 0))
    if max_total == 0:
        return

    usage = await redis.hgetall(f"llmapi:usage:{user_id}:{model}")
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    total = prompt + completion

    if total >= max_total:
        raise HTTPException(status_code=403, detail="Token limit exceeded")


async def record_usage(
    user_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    redis,
) -> None:
    key = f"llmapi:usage:{user_id}:{model}"
    if prompt_tokens:
        await redis.hincrby(key, "prompt_tokens", prompt_tokens)
    if completion_tokens:
        await redis.hincrby(key, "completion_tokens", completion_tokens)


def require_admin(user: User) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
