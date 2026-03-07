"""
Load test: simulate hundreds of requests per second across seeded users.

Seeds fresh users into Redis, then fires concurrent requests against the API.
Outputs structured JSON log grouped by user_id with per-user stats.

Modes:
  --mode gate   Hit GET /me — full auth + rate limit + token limit pipeline,
                zero LLM inference. Pure throughput of the auth gate layer.
                This is the default and the right mode for high-RPS testing.
  --mode auth   Hit LLM endpoints with max_tokens=4 — exercises auth pipeline
                plus minimal LLM work. Still bottlenecked by llama-server.
  --mode llm    Full LLM requests (/chat, /completion, /query). Limited by
                llama-server throughput; use lower RPS and higher timeout.

Usage:
    python load_test.py                                  # gate mode, 200 rps, 5s
    python load_test.py --rps 1000 --duration 10         # 1000 rps gate stress
    python load_test.py --mode auth --rps 50             # auth + minimal LLM
    python load_test.py --mode llm --rps 20 --duration 10
    python load_test.py --base-url http://host:8000
"""

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass, field

import httpx
import redis.asyncio as aioredis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REDIS_URL

# ---------------------------------------------------------------------------
# User definitions (mirrors seed_admin.py)
# ---------------------------------------------------------------------------

SEED_USERS = [
    {"name": "Admin", "is_admin": True},
    {"name": "Alice", "is_admin": False, "rate_limit": 60, "token_limit": 500_000},
    {"name": "Bob", "is_admin": False, "rate_limit": 30, "token_limit": 100_000},
    {"name": "Carol (Power User)", "is_admin": False, "rate_limit": 120},
]

# Requests are distributed with weights: heavier users get more traffic
USER_WEIGHTS = [1, 3, 2, 4]  # Admin=10%, Alice=30%, Bob=20%, Carol=40%

# Gate-mode endpoint — full auth pipeline, no LLM at all
GATE_ENDPOINTS = [
    {"method": "GET", "path": "/me", "body": None},
]

# Auth-mode endpoints — auth pipeline + minimal LLM work
AUTH_ENDPOINTS = [
    {"method": "POST", "path": "/chat", "body": {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 4, "temperature": 0.0,
    }},
    {"method": "POST", "path": "/completion", "body": {
        "prompt": "Hi", "max_tokens": 4, "temperature": 0.0,
    }},
    {"method": "POST", "path": "/query", "body": {
        "query": "1+1", "max_tokens": 4,
    }},
]

# LLM-mode endpoints — real inference
LLM_ENDPOINTS = [
    {"method": "POST", "path": "/chat", "body": {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16, "temperature": 0.1,
    }},
    {"method": "POST", "path": "/completion", "body": {
        "prompt": "Hi", "max_tokens": 16, "temperature": 0.1,
    }},
    {"method": "POST", "path": "/query", "body": {
        "query": "What is 1+1?", "max_tokens": 16,
    }},
]

ENDPOINTS_BY_MODE = {
    "gate": GATE_ENDPOINTS,
    "auth": AUTH_ENDPOINTS,
    "llm": LLM_ENDPOINTS,
}

DEFAULT_TIMEOUTS = {
    "gate": 5.0,
    "auth": 10.0,
    "llm": 120.0,
}


# ---------------------------------------------------------------------------
# Seed users into Redis
# ---------------------------------------------------------------------------

async def seed_users(redis_url: str) -> list[dict]:
    """Create fresh test users in Redis, return list with user_id + api_key."""
    r = aioredis.from_url(redis_url, decode_responses=True)
    users = []
    try:
        for cfg in SEED_USERS:
            user_id = secrets.token_hex(16)
            raw_key = f"llm-{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            key_prefix = raw_key[:12]

            await r.hset(f"llmapi:user:{user_id}", mapping={
                "name": cfg["name"],
                "is_admin": "true" if cfg["is_admin"] else "false",
                "created_at": str(int(time.time())),
            })
            await r.set(f"llmapi:apikey:{key_hash}", user_id)
            await r.sadd(f"llmapi:user:{user_id}:keys", f"{key_hash}:{key_prefix}")

            if cfg.get("rate_limit"):
                await r.hset(f"llmapi:ratelimit_config:{user_id}",
                             mapping={"requests_per_minute": str(cfg["rate_limit"])})
            if cfg.get("token_limit"):
                await r.hset(f"llmapi:limits:{user_id}:default",
                             mapping={"max_total_tokens": str(cfg["token_limit"])})

            users.append({
                "user_id": user_id,
                "name": cfg["name"],
                "api_key": raw_key,
                "rate_limit": cfg.get("rate_limit", 0),
                "token_limit": cfg.get("token_limit", 0),
            })
    finally:
        await r.aclose()
    return users


# ---------------------------------------------------------------------------
# Per-user stats tracker
# ---------------------------------------------------------------------------

@dataclass
class UserStats:
    user_id: str
    name: str
    rate_limit: int
    total: int = 0
    success: int = 0
    rate_limited: int = 0
    token_limited: int = 0
    auth_error: int = 0
    server_error: int = 0
    timeout: int = 0
    latencies: list[float] = field(default_factory=list)

    def record(self, status: int, latency: float):
        self.total += 1
        self.latencies.append(latency)
        if 200 <= status < 300:
            self.success += 1
        elif status == 429:
            self.rate_limited += 1
        elif status == 403:
            self.token_limited += 1
        elif status == 401:
            self.auth_error += 1
        elif status == 0:
            self.timeout += 1
        else:
            self.server_error += 1

    def to_dict(self) -> dict:
        lats = sorted(self.latencies) if self.latencies else [0]
        return {
            "user_id": self.user_id,
            "name": self.name,
            "rate_limit_rpm": self.rate_limit if self.rate_limit else "default",
            "total_requests": self.total,
            "success": self.success,
            "rate_limited_429": self.rate_limited,
            "token_limited_403": self.token_limited,
            "auth_error_401": self.auth_error,
            "server_error_5xx": self.server_error,
            "timeout": self.timeout,
            "latency_ms": {
                "min": round(lats[0] * 1000, 1),
                "median": round(lats[len(lats) // 2] * 1000, 1),
                "p95": round(lats[int(len(lats) * 0.95)] * 1000, 1),
                "p99": round(lats[int(len(lats) * 0.99)] * 1000, 1),
                "max": round(lats[-1] * 1000, 1),
            },
        }


# ---------------------------------------------------------------------------
# Request worker
# ---------------------------------------------------------------------------

async def send_request(
    client: httpx.AsyncClient,
    api_key: str,
    endpoint: dict,
    semaphore: asyncio.Semaphore,
    request_timeout: float,
) -> tuple[int, float]:
    """Fire a single request and return (status_code, latency_seconds)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    if endpoint.get("body") is not None:
        headers["Content-Type"] = "application/json"
    async with semaphore:
        t0 = time.monotonic()
        try:
            resp = await client.request(
                endpoint["method"],
                endpoint["path"],
                json=endpoint.get("body"),
                headers=headers,
                timeout=request_timeout,
            )
            return resp.status_code, time.monotonic() - t0
        except (httpx.TimeoutException, httpx.ConnectError):
            return 0, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Main load generator
# ---------------------------------------------------------------------------

async def run_load_test(
    base_url: str,
    rps: int,
    duration: int,
    redis_url: str,
    mode: str,
    request_timeout: float,
):
    print(f"Seeding test users into Redis ({redis_url})...", file=sys.stderr)
    users = await seed_users(redis_url)
    for u in users:
        print(
            f"  {u['name']:<25} id={u['user_id'][:12]}... "
            f"rate={u['rate_limit'] or 'default'}/min  "
            f"tokens={u['token_limit'] or 'unlimited'}",
            file=sys.stderr,
        )

    endpoints = ENDPOINTS_BY_MODE[mode]

    # Build weighted user pool
    pool: list[dict] = []
    for user, weight in zip(users, USER_WEIGHTS):
        pool.extend([user] * weight)

    stats: dict[str, UserStats] = {}
    for u in users:
        stats[u["user_id"]] = UserStats(
            user_id=u["user_id"],
            name=u["name"],
            rate_limit=u["rate_limit"],
        )

    total_requests = rps * duration
    interval = 1.0 / rps
    concurrency = min(rps * 2, 1000)
    semaphore = asyncio.Semaphore(concurrency)

    ep_names = sorted({ep["path"] for ep in endpoints})
    print(
        f"\nStarting load test: mode={mode}  {rps} rps x {duration}s = "
        f"{total_requests} requests\n"
        f"  concurrency={concurrency}  timeout={request_timeout}s  "
        f"endpoints={', '.join(ep_names)}",
        file=sys.stderr,
    )

    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks: list[asyncio.Task] = []
        t_start = time.monotonic()

        for i in range(total_requests):
            # Pick user round-robin from weighted pool
            user = pool[i % len(pool)]
            endpoint = endpoints[i % len(endpoints)]

            async def _do(u=user, ep=endpoint):
                status, latency = await send_request(
                    client, u["api_key"], ep, semaphore, request_timeout,
                )
                stats[u["user_id"]].record(status, latency)

            tasks.append(asyncio.create_task(_do()))

            # Pace requests
            elapsed = time.monotonic() - t_start
            expected = (i + 1) * interval
            if expected > elapsed:
                await asyncio.sleep(expected - elapsed)

            # Progress every 10%
            if (i + 1) % (total_requests // 10 or 1) == 0:
                pct = (i + 1) * 100 // total_requests
                done = sum(1 for t in tasks if t.done())
                print(
                    f"  [{pct:3d}%] {i + 1}/{total_requests} dispatched, "
                    f"{done} completed",
                    file=sys.stderr,
                )

        print("  Waiting for in-flight requests...", file=sys.stderr)
        await asyncio.gather(*tasks)

    wall_time = time.monotonic() - t_start

    # ---------------------------------------------------------------------------
    # Structured output
    # ---------------------------------------------------------------------------

    total_success = sum(s.success for s in stats.values())
    total_429 = sum(s.rate_limited for s in stats.values())
    total_403 = sum(s.token_limited for s in stats.values())
    total_timeout = sum(s.timeout for s in stats.values())
    all_lats = sorted(lat for s in stats.values() for lat in s.latencies)

    output = {
        "summary": {
            "mode": mode,
            "target_rps": rps,
            "duration_seconds": duration,
            "total_requests": total_requests,
            "wall_time_seconds": round(wall_time, 2),
            "actual_rps": round(total_requests / wall_time, 1),
            "total_success": total_success,
            "total_rate_limited_429": total_429,
            "total_token_limited_403": total_403,
            "total_timeout": total_timeout,
            "total_errors": total_requests - total_success - total_429 - total_403 - total_timeout,
            "latency_ms": {
                "min": round(all_lats[0] * 1000, 1) if all_lats else 0,
                "median": round(all_lats[len(all_lats) // 2] * 1000, 1) if all_lats else 0,
                "p95": round(all_lats[int(len(all_lats) * 0.95)] * 1000, 1) if all_lats else 0,
                "p99": round(all_lats[int(len(all_lats) * 0.99)] * 1000, 1) if all_lats else 0,
                "max": round(all_lats[-1] * 1000, 1) if all_lats else 0,
            },
        },
        "users": [stats[u["user_id"]].to_dict() for u in users],
    }

    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load test the LLM API with seeded users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  gate    Hit GET /me — full auth + rate limit + token limit checks, zero LLM.
          Pure request-handling throughput. Best for high-RPS stress testing.
  auth    Hit LLM endpoints with max_tokens=4 — auth pipeline + minimal LLM.
  llm     Full LLM inference — use lower --rps and higher --timeout.

examples:
  python load_test.py                                  # gate mode, 200 rps, 5s
  python load_test.py --rps 1000 --duration 10         # 1000 rps gate stress
  python load_test.py --mode auth --rps 50             # auth + minimal LLM
  python load_test.py --mode llm --rps 20 --duration 10
  python load_test.py --mode llm --timeout 180         # long LLM timeout
""",
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--rps", type=int, default=200, help="Requests per second (default: 200)")
    parser.add_argument("--duration", type=int, default=5, help="Test duration in seconds (default: 5)")
    parser.add_argument("--redis-url", default=REDIS_URL, help="Redis URL for seeding users")
    parser.add_argument(
        "--mode",
        choices=["gate", "auth", "llm"],
        default="gate",
        help="Test mode: 'gate' for pure auth/rate-limit stress (default), "
             "'auth' for auth + minimal LLM, 'llm' for full inference",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: 5 for gate, 10 for auth, 120 for llm)",
    )
    args = parser.parse_args()

    if args.timeout is None:
        args.timeout = DEFAULT_TIMEOUTS[args.mode]

    asyncio.run(run_load_test(
        args.base_url,
        args.rps,
        args.duration,
        args.redis_url,
        args.mode,
        args.timeout,
    ))


if __name__ == "__main__":
    main()
