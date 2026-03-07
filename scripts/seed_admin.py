"""Bootstrap CLI: create initial users and API keys."""

import asyncio
import hashlib
import os
import secrets
import sys
import time

import redis.asyncio as aioredis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REDIS_URL

SEED_USERS = [
    {"name": "Admin", "is_admin": True},
    {"name": "Alice", "is_admin": False, "rate_limit": 60, "rate_limit_hour": 500, "rate_limit_day": 5000, "token_limit": 500_000},
    {"name": "Bob", "is_admin": False, "rate_limit": 30, "rate_limit_hour": 200, "rate_limit_day": 2000, "token_limit": 100_000},
    {"name": "Carol (Power User)", "is_admin": False, "rate_limit": 120, "rate_limit_hour": 2000},
]


async def create_user(r, *, name: str, is_admin: bool, rate_limit: int = 0, rate_limit_hour: int = 0, rate_limit_day: int = 0, token_limit: int = 0):
    user_id = secrets.token_hex(16)
    raw_key = f"llm-{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:12]

    await r.hset(
        f"llmapi:user:{user_id}",
        mapping={
            "name": name,
            "is_admin": "true" if is_admin else "false",
            "created_at": str(int(time.time())),
        },
    )
    await r.set(f"llmapi:apikey:{key_hash}", user_id)
    await r.sadd(f"llmapi:user:{user_id}:keys", f"{key_hash}:{key_prefix}")

    rl_mapping = {}
    if rate_limit:
        rl_mapping["requests_per_minute"] = str(rate_limit)
    if rate_limit_hour:
        rl_mapping["requests_per_hour"] = str(rate_limit_hour)
    if rate_limit_day:
        rl_mapping["requests_per_day"] = str(rate_limit_day)
    if rl_mapping:
        await r.hset(f"llmapi:ratelimit_config:{user_id}", mapping=rl_mapping)

    if token_limit:
        await r.hset(
            f"llmapi:limits:{user_id}:default",
            mapping={"max_total_tokens": str(token_limit)},
        )

    return user_id, raw_key, rate_limit, token_limit


async def main():
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        print("Seeding users...\n")
        print(f"{'Name':<25} {'Role':<8} {'Rate':<10} {'Token Limit':<12} API Key")
        print("-" * 120)

        for user_cfg in SEED_USERS:
            user_id, raw_key, rate, tokens = await create_user(r, **user_cfg)
            role = "admin" if user_cfg["is_admin"] else "user"
            rate_str = f"{rate}/min" if rate else "default"
            token_str = f"{tokens:,}" if tokens else "unlimited"
            print(f"{user_cfg['name']:<25} {role:<8} {rate_str:<10} {token_str:<12} {raw_key}")

        print(f"\nDone — {len(SEED_USERS)} users created.")
        print("Save these API keys now — they cannot be retrieved later.")
    finally:
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
