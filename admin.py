"""Admin endpoints for user/key/limit management."""

import hashlib
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import User, get_current_user, get_redis, require_admin

router = APIRouter(prefix="/admin")


class CreateUserRequest(BaseModel):
    name: str = Field(..., description="User display name")
    is_admin: bool = Field(default=False, description="Grant admin privileges")


class CreateUserResponse(BaseModel):
    user_id: str


class UserInfo(BaseModel):
    user_id: str
    name: str
    is_admin: bool
    created_at: str


class CreateKeyResponse(BaseModel):
    api_key: str
    key_prefix: str


class SetTokenLimitRequest(BaseModel):
    max_total_tokens: int = Field(..., ge=0, description="0 = unlimited")


class SetRateLimitRequest(BaseModel):
    requests_per_minute: int = Field(..., ge=1)


class UsageInfo(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@router.post("/users", response_model=CreateUserResponse)
async def create_user(
    request: CreateUserRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    user_id = secrets.token_hex(16)
    await redis.hset(
        f"llmapi:user:{user_id}",
        mapping={
            "name": request.name,
            "is_admin": "true" if request.is_admin else "false",
            "created_at": str(int(time.time())),
        },
    )
    return CreateUserResponse(user_id=user_id)


@router.get("/users", response_model=list[UserInfo])
async def list_users(
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    cursor = "0"
    users = []
    while True:
        cursor, keys = await redis.scan(cursor=cursor, match="llmapi:user:*", count=100)
        for key in keys:
            if ":keys" in key:
                continue
            uid = key.split("llmapi:user:")[1]
            data = await redis.hgetall(key)
            users.append(UserInfo(
                user_id=uid,
                name=data.get("name", ""),
                is_admin=data.get("is_admin", "false") == "true",
                created_at=data.get("created_at", ""),
            ))
        if cursor == "0" or cursor == 0:
            break
    return users


@router.get("/users/{user_id}", response_model=UserInfo)
async def get_user(
    user_id: str,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    data = await redis.hgetall(f"llmapi:user:{user_id}")
    if not data:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfo(
        user_id=user_id,
        name=data.get("name", ""),
        is_admin=data.get("is_admin", "false") == "true",
        created_at=data.get("created_at", ""),
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    exists = await redis.exists(f"llmapi:user:{user_id}")
    if not exists:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete all API keys
    key_members = await redis.smembers(f"llmapi:user:{user_id}:keys")
    for member in key_members:
        key_hash = member.split(":")[0]
        await redis.delete(f"llmapi:apikey:{key_hash}")
    await redis.delete(f"llmapi:user:{user_id}:keys")

    # Delete rate limit and usage data
    await redis.delete(f"llmapi:ratelimit:{user_id}")
    await redis.delete(f"llmapi:ratelimit_config:{user_id}")

    # Clean up usage/limits keys
    cursor = "0"
    while True:
        cursor, keys = await redis.scan(
            cursor=cursor,
            match=f"llmapi:usage:{user_id}:*",
            count=100,
        )
        for key in keys:
            await redis.delete(key)
        if cursor == "0" or cursor == 0:
            break
    cursor = "0"
    while True:
        cursor, keys = await redis.scan(
            cursor=cursor,
            match=f"llmapi:limits:{user_id}:*",
            count=100,
        )
        for key in keys:
            await redis.delete(key)
        if cursor == "0" or cursor == 0:
            break

    await redis.delete(f"llmapi:user:{user_id}")
    return {"message": f"User {user_id} deleted"}


@router.post("/users/{user_id}/keys", response_model=CreateKeyResponse)
async def create_api_key(
    user_id: str,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    exists = await redis.exists(f"llmapi:user:{user_id}")
    if not exists:
        raise HTTPException(status_code=404, detail="User not found")

    raw_key = f"llm-{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:12]

    await redis.set(f"llmapi:apikey:{key_hash}", user_id)
    await redis.sadd(f"llmapi:user:{user_id}:keys", f"{key_hash}:{key_prefix}")

    return CreateKeyResponse(api_key=raw_key, key_prefix=key_prefix)


@router.delete("/users/{user_id}/keys/{key_prefix}")
async def revoke_api_key(
    user_id: str,
    key_prefix: str,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    members = await redis.smembers(f"llmapi:user:{user_id}:keys")
    found = None
    for member in members:
        if member.endswith(f":{key_prefix}"):
            found = member
            break
    if not found:
        raise HTTPException(status_code=404, detail="Key not found")

    key_hash = found.split(":")[0]
    await redis.delete(f"llmapi:apikey:{key_hash}")
    await redis.srem(f"llmapi:user:{user_id}:keys", found)
    return {"message": "Key revoked"}


@router.put("/users/{user_id}/limits/{model}")
async def set_token_limit(
    user_id: str,
    model: str,
    request: SetTokenLimitRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    exists = await redis.exists(f"llmapi:user:{user_id}")
    if not exists:
        raise HTTPException(status_code=404, detail="User not found")

    if request.max_total_tokens == 0:
        await redis.delete(f"llmapi:limits:{user_id}:{model}")
    else:
        await redis.hset(
            f"llmapi:limits:{user_id}:{model}",
            mapping={"max_total_tokens": str(request.max_total_tokens)},
        )
    return {"message": f"Token limit set for {model}"}


@router.put("/users/{user_id}/rate-limit")
async def set_rate_limit(
    user_id: str,
    request: SetRateLimitRequest,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    exists = await redis.exists(f"llmapi:user:{user_id}")
    if not exists:
        raise HTTPException(status_code=404, detail="User not found")

    await redis.hset(
        f"llmapi:ratelimit_config:{user_id}",
        mapping={"requests_per_minute": str(request.requests_per_minute)},
    )
    return {"message": "Rate limit updated"}


@router.get("/users/{user_id}/usage", response_model=list[UsageInfo])
async def get_usage(
    user_id: str,
    user: User = Depends(get_current_user),
    redis=Depends(get_redis),
):
    require_admin(user)
    exists = await redis.exists(f"llmapi:user:{user_id}")
    if not exists:
        raise HTTPException(status_code=404, detail="User not found")

    usages = []
    cursor = "0"
    while True:
        cursor, keys = await redis.scan(
            cursor=cursor,
            match=f"llmapi:usage:{user_id}:*",
            count=100,
        )
        for key in keys:
            model = key.split(f"llmapi:usage:{user_id}:")[1]
            data = await redis.hgetall(key)
            prompt = int(data.get("prompt_tokens", 0))
            completion = int(data.get("completion_tokens", 0))
            usages.append(UsageInfo(
                model=model,
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            ))
        if cursor == "0" or cursor == 0:
            break
    return usages
