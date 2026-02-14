import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_httpx():
    """Mock httpx.AsyncClient to avoid real calls to llama-server."""
    with patch("api.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
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
