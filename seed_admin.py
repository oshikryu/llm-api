"""Bootstrap CLI: create the first admin user and API key."""

import asyncio
import hashlib
import secrets
import time

import redis.asyncio as aioredis

from config import REDIS_URL


async def main():
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        user_id = secrets.token_hex(16)
        raw_key = f"llm-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:12]

        await r.hset(
            f"llmapi:user:{user_id}",
            mapping={
                "name": "Admin",
                "is_admin": "true",
                "created_at": str(int(time.time())),
            },
        )
        await r.set(f"llmapi:apikey:{key_hash}", user_id)
        await r.sadd(f"llmapi:user:{user_id}:keys", f"{key_hash}:{key_prefix}")

        print(f"Admin user created: {user_id}")
        print(f"API key (save this — shown only once): {raw_key}")
    finally:
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
