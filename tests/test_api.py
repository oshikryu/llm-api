"""Tests for api.py endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tests.conftest import make_chat_response, make_completion_response


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_llama_ok(self, client, mock_httpx):
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "ok"}),
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api"] == "ok"
        assert data["llama_server"]["status"] == "ok"

    def test_health_llama_unreachable(self, client, mock_httpx):
        mock_httpx.get.side_effect = Exception("connection refused")
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api"] == "ok"
        assert data["llama_server"]["status"] == "unreachable"


# ---------------------------------------------------------------------------
# POST /completion
# ---------------------------------------------------------------------------


class TestCompletion:
    def test_basic_completion(self, client, mock_httpx):
        mock_httpx.post.return_value = make_completion_response(
            content="Machine learning is...", tokens_predicted=20, tokens_evaluated=8
        )
        resp = client.post("/completion", json={"prompt": "What is ML?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Machine learning is..."
        assert data["tokens_generated"] == 20
        assert data["tokens_prompt"] == 8

    def test_completion_with_params(self, client, mock_httpx):
        mock_httpx.post.return_value = make_completion_response()
        resp = client.post(
            "/completion",
            json={
                "prompt": "Hello",
                "max_tokens": 256,
                "temperature": 0.5,
                "top_p": 0.8,
                "stop": ["\n"],
            },
        )
        assert resp.status_code == 200
        call_payload = mock_httpx.post.call_args[1]["json"]
        assert call_payload["n_predict"] == 256
        assert call_payload["temperature"] == 0.5
        assert call_payload["top_p"] == 0.8
        assert call_payload["stop"] == ["\n"]

    def test_completion_timeout(self, client, mock_httpx):
        mock_httpx.post.side_effect = httpx.TimeoutException("timed out")
        resp = client.post("/completion", json={"prompt": "Hello"})
        assert resp.status_code == 504

    def test_completion_server_error(self, client, mock_httpx):
        mock_httpx.post.side_effect = httpx.HTTPStatusError(
            "error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        resp = client.post("/completion", json={"prompt": "Hello"})
        assert resp.status_code == 500

    def test_completion_validation_error(self, client):
        resp = client.post("/completion", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------


class TestChat:
    def test_basic_chat(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response(
            content="Hello! How can I help?",
            completion_tokens=15,
            prompt_tokens=10,
        )
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello! How can I help?"
        assert data["tokens_generated"] == 15
        assert data["tokens_prompt"] == 10

    def test_chat_with_system_message(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response()
        resp = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )
        assert resp.status_code == 200
        call_payload = mock_httpx.post.call_args[1]["json"]
        assert len(call_payload["messages"]) == 2
        assert call_payload["messages"][0]["role"] == "system"

    def test_chat_continue_until_done_stops_on_stop(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response(finish_reason="stop")
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "continue_until_done": True,
            },
        )
        assert resp.status_code == 200
        # Should only call once since finish_reason is "stop"
        assert mock_httpx.post.call_count == 1

    def test_chat_continue_until_done_concatenates(self, client, mock_httpx):
        """When finish_reason is 'length', should continue generating."""
        mock_httpx.post.side_effect = [
            make_chat_response(
                content="First part ",
                finish_reason="length",
                completion_tokens=512,
            ),
            make_chat_response(
                content="second part.",
                finish_reason="stop",
                completion_tokens=10,
            ),
        ]
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "continue_until_done": True,
                "max_tokens": 512,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "First part " in data["message"]["content"]
        assert "second part." in data["message"]["content"]
        assert mock_httpx.post.call_count == 2

    def test_chat_timeout(self, client, mock_httpx):
        mock_httpx.post.side_effect = httpx.TimeoutException("timeout")
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 504

    def test_chat_missing_messages(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_markdown(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response(content="# Answer\nHere it is.")
        resp = client.post("/query", json={"query": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is Python?"
        assert data["format"] == "markdown"
        assert data["response"] == "# Answer\nHere it is."
        assert data["structured"] is None

    def test_query_json_format(self, client, mock_httpx):
        json_content = '{"summary": "Python is a language", "confidence": "high"}'
        mock_httpx.post.return_value = make_chat_response(content=json_content)
        resp = client.post(
            "/query",
            json={"query": "What is Python?", "output_format": "json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "json"
        assert data["structured"]["summary"] == "Python is a language"

    def test_query_json_in_code_block(self, client, mock_httpx):
        json_content = '```json\n{"summary": "test"}\n```'
        mock_httpx.post.return_value = make_chat_response(content=json_content)
        resp = client.post(
            "/query",
            json={"query": "test", "output_format": "json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["structured"]["summary"] == "test"

    def test_query_plain_format(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response(content="Just text.")
        resp = client.post(
            "/query",
            json={"query": "test", "output_format": "plain"},
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "plain"

    def test_query_custom_system_prompt(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response(content="custom response")
        resp = client.post(
            "/query",
            json={"query": "test", "system_prompt": "You are a pirate."},
        )
        assert resp.status_code == 200
        call_payload = mock_httpx.post.call_args[1]["json"]
        assert call_payload["messages"][0]["content"] == "You are a pirate."


# ---------------------------------------------------------------------------
# GET /query
# ---------------------------------------------------------------------------


class TestQueryGet:
    def test_query_get(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response(content="Answer here.")
        resp = client.get("/query", params={"q": "What is AI?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is AI?"

    def test_query_get_with_params(self, client, mock_httpx):
        mock_httpx.post.return_value = make_chat_response(content="{}")
        resp = client.get(
            "/query",
            params={
                "q": "test",
                "format": "json",
                "max_tokens": 256,
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "json"


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze_with_context(self, client, mock_httpx):
        mock_httpx.post.return_value = make_completion_response(content="Summary here.")
        resp = client.post(
            "/analyze",
            params={"prompt": "Summarize", "context": "Some long text."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Summary here."
        # Verify the prompt includes context
        call_payload = mock_httpx.post.call_args[1]["json"]
        assert "Some long text." in call_payload["prompt"]
        assert "Summarize" in call_payload["prompt"]

    def test_analyze_without_context(self, client, mock_httpx):
        mock_httpx.post.return_value = make_completion_response(content="Done.")
        resp = client.post("/analyze", params={"prompt": "Do something"})
        assert resp.status_code == 200
        call_payload = mock_httpx.post.call_args[1]["json"]
        assert call_payload["prompt"] == "Do something"


# ---------------------------------------------------------------------------
# _looks_complete helper
# ---------------------------------------------------------------------------


class TestLooksComplete:
    def test_few_tokens(self):
        from api import _looks_complete

        assert _looks_complete("Short answer.", 50, 512) is True

    def test_ends_with_period(self):
        from api import _looks_complete

        assert _looks_complete("This is a sentence.", 400, 512) is True

    def test_ends_with_code_block(self):
        from api import _looks_complete

        assert _looks_complete("some code\n```", 400, 512) is True

    def test_ends_with_colon_incomplete(self):
        from api import _looks_complete

        assert _looks_complete("The items are:", 400, 512) is False

    def test_ends_with_dash_incomplete(self):
        from api import _looks_complete

        assert _looks_complete("- item one\n-", 400, 512) is False
