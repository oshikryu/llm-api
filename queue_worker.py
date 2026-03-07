"""arq worker configuration. Run with: arq queue_worker.WorkerSettings"""

import httpx
import redis.asyncio as aioredis
from arq.connections import RedisSettings

from config import LLAMA_SERVER_URL, LLAMA_PARALLEL_SLOTS, REDIS_URL
from queue_tasks import process_chat_completion, process_completion


async def startup(ctx: dict):
    ctx["http_client"] = httpx.AsyncClient(base_url=LLAMA_SERVER_URL)
    try:
        ctx["redis"] = aioredis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        ctx["redis"] = None


async def shutdown(ctx: dict):
    await ctx["http_client"].aclose()
    if ctx.get("redis") is not None:
        await ctx["redis"].aclose()


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    functions = [process_chat_completion, process_completion]
    max_jobs = LLAMA_PARALLEL_SLOTS
    on_startup = startup
    on_shutdown = shutdown
