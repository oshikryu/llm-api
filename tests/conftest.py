import asyncio

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def disable_auth():
    """Disable auth for all tests by default."""
    with patch("auth.AUTH_ENABLED", False):
        yield


@pytest.fixture()
def mock_redis():
    """Provide a mock Redis client on app.state.redis."""
    mock_r = AsyncMock()
    mock_r.hgetall.return_value = {}
    mock_r.get.return_value = None
    mock_r.zcard.return_value = 0
    mock_r.zcount.return_value = 0
    mock_r.hincrby.return_value = 0
    mock_r.llen.return_value = 0
    mock_r.lrange.return_value = []
    from api import app
    app.state.redis = mock_r
    yield mock_r


@pytest.fixture()
def mock_httpx(mock_redis):
    """Mock the shared http_client and llm_semaphore in api module."""
    mock_client = AsyncMock()
    with (
        patch("api.http_client", mock_client),
        patch("api.llm_semaphore", asyncio.Semaphore(4)),
    ):
        yield mock_client


@pytest.fixture()
def client(mock_redis):
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
