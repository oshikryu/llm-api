"""Generate token usage for seeded users by making real API requests."""

import asyncio
import sys

import httpx

BASE_URL = "http://localhost:8000"

USERS = [
    {
        "name": "Alice",
        "key": "llm-s5ReOK6bcYtoQuGrCyG5UvWp4WZ1LcL6iHNfbSuQCZQ",
        "requests": [
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "What is Python?"}], "max_tokens": 64}},
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "Explain HTTP in one sentence."}], "max_tokens": 64}},
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "What is REST?"}], "max_tokens": 64}},
            {"endpoint": "/completion", "body": {"prompt": "The capital of France is", "max_tokens": 32}},
            {"endpoint": "/completion", "body": {"prompt": "Machine learning is", "max_tokens": 64}},
        ],
    },
    {
        "name": "Bob",
        "key": "llm-LwXJHYWAbvJ4nVqK7gTGNcCLB4N_neMzkbu3fDZXIz0",
        "requests": [
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 32}},
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "What is Docker?"}], "max_tokens": 64}},
            {"endpoint": "/completion", "body": {"prompt": "List three colors:", "max_tokens": 32}},
        ],
    },
    {
        "name": "Carol",
        "key": "llm-A9d_U44so-D-9lND5EWXBkvXoKuVw0DNUz-UD7GLPqI",
        "requests": [
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "Explain recursion briefly."}], "max_tokens": 128}},
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "What are design patterns?"}], "max_tokens": 128}},
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "Compare TCP and UDP."}], "max_tokens": 128}},
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "What is Kubernetes?"}], "max_tokens": 128}},
            {"endpoint": "/completion", "body": {"prompt": "The benefits of microservices are", "max_tokens": 128}},
            {"endpoint": "/completion", "body": {"prompt": "Functional programming is", "max_tokens": 64}},
            {"endpoint": "/completion", "body": {"prompt": "A binary tree is", "max_tokens": 64}},
        ],
    },
    {
        "name": "Admin",
        "key": "llm-bX3ZMo5tSDexqFi2VUO_NUn_Oj57l-LUBhYkAFXvV6k",
        "requests": [
            {"endpoint": "/chat", "body": {"messages": [{"role": "user", "content": "What is FastAPI?"}], "max_tokens": 64}},
            {"endpoint": "/completion", "body": {"prompt": "Redis is a", "max_tokens": 32}},
        ],
    },
]


async def send_request(client: httpx.AsyncClient, user_name: str, api_key: str, req: dict) -> dict:
    endpoint = req["endpoint"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = await client.post(f"{BASE_URL}{endpoint}", json=req["body"], headers=headers, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            prompt_tok = data.get("tokens_prompt", 0)
            gen_tok = data.get("tokens_generated", 0)
            print(f"  {user_name:<8} {endpoint:<14} {resp.status_code}  prompt={prompt_tok:>5}  generated={gen_tok:>5}")
            return {"ok": True, "prompt": prompt_tok, "generated": gen_tok}
        else:
            print(f"  {user_name:<8} {endpoint:<14} {resp.status_code}  {resp.text[:80]}", file=sys.stderr)
            return {"ok": False}
    except Exception as e:
        print(f"  {user_name:<8} {endpoint:<14} ERROR  {e}", file=sys.stderr)
        return {"ok": False}


async def main():
    print("Generating usage for seeded users...\n")
    async with httpx.AsyncClient() as client:
        for user in USERS:
            name = user["name"]
            key = user["key"]
            total_prompt = 0
            total_gen = 0
            success = 0

            print(f"--- {name} ({len(user['requests'])} requests) ---")
            for req in user["requests"]:
                result = await send_request(client, name, key, req)
                if result["ok"]:
                    success += 1
                    total_prompt += result["prompt"]
                    total_gen += result["generated"]

            print(f"  Summary: {success}/{len(user['requests'])} ok, "
                  f"prompt_tokens={total_prompt}, generated_tokens={total_gen}\n")

    print("Done. Check usage with:")
    print("  curl -H 'Authorization: Bearer <admin_key>' http://localhost:8000/admin/users")
    print("  curl -H 'Authorization: Bearer <user_key>' http://localhost:8000/billing")


if __name__ == "__main__":
    asyncio.run(main())
