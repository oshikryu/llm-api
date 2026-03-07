"""Tests for async queue endpoints."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
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
    from api import app
    return TestClient(app)


class TestAsyncChatEndpoint:
    def test_chat_async_returns_job_id(self, client, mock_httpx):
        mock_pool = AsyncMock()
        with patch("api.arq_pool", mock_pool):
            resp = client.post(
                "/chat/async",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        mock_pool.enqueue_job.assert_called_once()

    def test_chat_async_no_redis(self, client, mock_httpx):
        with patch("api.arq_pool", None):
            resp = client.post(
                "/chat/async",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        assert resp.status_code == 503


class TestAsyncCompletionEndpoint:
    def test_completion_async_returns_job_id(self, client, mock_httpx):
        mock_pool = AsyncMock()
        with patch("api.arq_pool", mock_pool):
            resp = client.post(
                "/completion/async",
                json={"prompt": "Hello"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_completion_async_no_redis(self, client, mock_httpx):
        with patch("api.arq_pool", None):
            resp = client.post(
                "/completion/async",
                json={"prompt": "Hello"},
            )
        assert resp.status_code == 503


class TestJobStatus:
    def test_job_not_found(self, client, mock_httpx):
        mock_pool = AsyncMock()
        with patch("api.arq_pool", mock_pool):
            mock_job_cls = MagicMock()
            mock_job_instance = AsyncMock()
            mock_job_instance.info.return_value = None
            mock_job_cls.return_value = mock_job_instance
            with patch("api.Job", mock_job_cls, create=True):
                # The endpoint imports Job from arq.jobs at call time
                with patch("arq.jobs.Job") as mock_arq_job:
                    mock_arq_job.return_value = mock_job_instance
                    resp = client.get("/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_job_complete(self, client, mock_httpx):
        mock_pool = AsyncMock()
        mock_info = MagicMock()
        mock_info.status = "complete"
        mock_info.result = {"message": {"role": "assistant", "content": "Hi"}}

        with patch("api.arq_pool", mock_pool):
            with patch("arq.jobs.Job") as mock_arq_job:
                mock_job_instance = AsyncMock()
                mock_job_instance.info.return_value = mock_info
                mock_arq_job.return_value = mock_job_instance
                resp = client.get("/jobs/some-job-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["result"]["message"]["content"] == "Hi"

    def test_job_status_no_redis(self, client, mock_httpx):
        with patch("api.arq_pool", None):
            resp = client.get("/jobs/some-id")
        assert resp.status_code == 503
