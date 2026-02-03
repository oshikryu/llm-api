"""
Example client for the Local LLM API.

Usage:
    python client.py "What are the O-1A visa criteria?"
"""

import argparse
import json
import sys
import httpx

API_URL = "http://localhost:8000"


def completion(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Send a completion request to the API."""
    response = httpx.post(
        f"{API_URL}/completion",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["text"]


def chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.7, continue_until_done: bool = True) -> str:
    """Send a chat request to the API (non-streaming)."""
    response = httpx.post(
        f"{API_URL}/chat",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "continue_until_done": continue_until_done,
        },
        timeout=600.0 if continue_until_done else 120.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def chat_stream(messages: list[dict], max_tokens: int = 512, temperature: float = 0.7, continue_until_done: bool = True) -> None:
    """Send a streaming chat request, printing output in real-time."""
    with httpx.stream(
        "POST",
        f"{API_URL}/chat/stream",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "continue_until_done": continue_until_done,
        },
        timeout=600.0,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            sys.stdout.write(chunk.decode("utf-8"))
            sys.stdout.flush()
    print()  # Newline at end


def query(
    q: str,
    system_prompt: str | None = None,
    output_format: str = "markdown",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    """Simple query - returns complete structured response."""
    payload = {
        "query": q,
        "output_format": output_format,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt

    response = httpx.post(
        f"{API_URL}/query",
        json=payload,
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()


def health_check() -> dict:
    """Check API and LLM server health."""
    response = httpx.get(f"{API_URL}/health", timeout=5.0)
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the local LLM API")
    parser.add_argument("prompt", nargs="?", help="Prompt to send to the LLM")
    parser.add_argument("--chat", action="store_true", help="Use chat mode (streaming)")
    parser.add_argument("--completion", action="store_true", help="Use raw completion mode")
    parser.add_argument("--system", type=str, help="System prompt for query mode")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "json", "plain"], help="Output format")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--no-continue", dest="continue_until_done", action="store_false", help="Stop at max_tokens instead of continuing")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming output")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON response")
    parser.add_argument("--health", action="store_true", help="Check health status")

    args = parser.parse_args()

    if args.health:
        print(health_check())
    elif args.prompt:
        if args.completion:
            # Raw completion mode
            result = completion(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(result)
        elif args.chat:
            # Streaming chat mode
            if args.stream:
                chat_stream(
                    [{"role": "user", "content": args.prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    continue_until_done=args.continue_until_done,
                )
            else:
                result = chat(
                    [{"role": "user", "content": args.prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    continue_until_done=args.continue_until_done,
                )
                print(result)
        else:
            # Default: simple query mode (complete response)
            result = query(
                args.prompt,
                system_prompt=args.system,
                output_format=args.format,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            if args.raw:
                print(json.dumps(result, indent=2))
            elif args.format == "json" and result.get("structured"):
                print(json.dumps(result["structured"], indent=2))
            else:
                print(result["response"])
    else:
        parser.print_help()
