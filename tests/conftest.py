import asyncio

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_httpx():
    """Mock the shared http_client and llm_semaphore in api module."""
    mock_client = AsyncMock()
    with (
        patch("api.http_client", mock_client),
        patch("api.llm_semaphore", asyncio.Semaphore(4)),
    ):
        yield mock_client


@pytest.fixture()
def client():
    """TestClient for the FastAPI app."""
    from api import app
    return TestClient(app)


def make_completion_response(content="Hello!", tokens_predicted=10, tokens_evaluated=5):
    """Helper to build a mock llama-server /completion response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "content": content,
        "tokens_predicted": tokens_predicted,
        "tokens_evaluated": tokens_evaluated,
    }
    resp.raise_for_status = MagicMock()
    return resp


def make_chat_response(
    content="Hi there!",
    finish_reason="stop",
    completion_tokens=10,
    prompt_tokens=5,
):
    """Helper to build a mock llama-server /v1/chat/completions response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
        },
    }
    resp.raise_for_status = MagicMock()
    return resp
