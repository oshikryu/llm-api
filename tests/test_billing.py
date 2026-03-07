"""Tests for billing/usage endpoints: /usage, /usage/{model}, /limits, /billing."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from auth import User


@pytest.fixture()
def billing_user():
    return User(user_id="user-123", name="Alice", is_admin=False)


@pytest.fixture()
def billing_client(mock_redis, billing_user):
    from api import app
    from auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: billing_user
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_current_user, None)


def _mock_scan(keys_by_pattern: dict[str, list[str]]):
    """Return an async side_effect for redis.scan that returns matching keys."""
    async def scan_side_effect(cursor="0", match="", count=100):
        return ("0", keys_by_pattern.get(match, []))
    return scan_side_effect


class TestGetUsage:
    def test_usage_no_models(self, billing_client, mock_redis):
        mock_redis.scan = AsyncMock(side_effect=_mock_scan({}))
        resp = billing_client.get("/usage")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_usage_single_model(self, billing_client, mock_redis):
        mock_redis.scan = AsyncMock(
            side_effect=_mock_scan({"llmapi:usage:user-123:*": ["llmapi:usage:user-123:default"]})
        )
        mock_redis.hgetall = AsyncMock(return_value={"prompt_tokens": "100", "completion_tokens": "50"})
        resp = billing_client.get("/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model"] == "default"
        assert data[0]["prompt_tokens"] == 100
        assert data[0]["completion_tokens"] == 50
        assert data[0]["total_tokens"] == 150

    def test_usage_multiple_models(self, billing_client, mock_redis):
        mock_redis.scan = AsyncMock(
            side_effect=_mock_scan({
                "llmapi:usage:user-123:*": [
                    "llmapi:usage:user-123:default",
                    "llmapi:usage:user-123:llama-3",
                ]
            })
        )
        call_count = 0
        responses = [
            {"prompt_tokens": "100", "completion_tokens": "50"},
            {"prompt_tokens": "200", "completion_tokens": "300"},
        ]

        async def hgetall_side(key):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return responses[idx]

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side)
        resp = billing_client.get("/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


class TestGetUsageByModel:
    def test_usage_existing_model(self, billing_client, mock_redis):
        mock_redis.hgetall = AsyncMock(return_value={"prompt_tokens": "500", "completion_tokens": "200"})
        resp = billing_client.get("/usage/default")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "default"
        assert data["prompt_tokens"] == 500
        assert data["completion_tokens"] == 200
        assert data["total_tokens"] == 700

    def test_usage_nonexistent_model(self, billing_client, mock_redis):
        mock_redis.hgetall = AsyncMock(return_value={})
        resp = billing_client.get("/usage/unknown-model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "unknown-model"
        assert data["prompt_tokens"] == 0
        assert data["completion_tokens"] == 0
        assert data["total_tokens"] == 0


class TestGetLimits:
    def test_limits_no_models(self, billing_client, mock_redis):
        mock_redis.scan = AsyncMock(side_effect=_mock_scan({}))
        resp = billing_client.get("/limits")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_limits_with_limit_set(self, billing_client, mock_redis):
        mock_redis.scan = AsyncMock(
            side_effect=_mock_scan({
                "llmapi:limits:user-123:*": ["llmapi:limits:user-123:default"],
                "llmapi:usage:user-123:*": ["llmapi:usage:user-123:default"],
            })
        )

        async def hgetall_side(key):
            if "limits" in key:
                return {"max_total_tokens": "500000"}
            return {"prompt_tokens": "100000", "completion_tokens": "50000"}

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side)
        resp = billing_client.get("/limits")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert item["model"] == "default"
        assert item["max_total_tokens"] == 500000
        assert item["current_total_tokens"] == 150000
        assert item["remaining_tokens"] == 350000
        assert item["usage_percent"] == 30.0

    def test_limits_unlimited(self, billing_client, mock_redis):
        """Model with usage but no limit set (max_total_tokens=0)."""
        mock_redis.scan = AsyncMock(
            side_effect=_mock_scan({
                "llmapi:limits:user-123:*": [],
                "llmapi:usage:user-123:*": ["llmapi:usage:user-123:default"],
            })
        )

        async def hgetall_side(key):
            if "limits" in key:
                return {}
            return {"prompt_tokens": "1000", "completion_tokens": "500"}

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side)
        resp = billing_client.get("/limits")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert item["max_total_tokens"] == 0
        assert item["remaining_tokens"] is None
        assert item["usage_percent"] is None


class TestGetBilling:
    def test_billing_full(self, billing_client, mock_redis):
        mock_redis.scan = AsyncMock(
            side_effect=_mock_scan({
                "llmapi:usage:user-123:*": ["llmapi:usage:user-123:default"],
                "llmapi:limits:user-123:*": ["llmapi:limits:user-123:default"],
            })
        )

        async def hgetall_side(key):
            if "ratelimit_config" in key:
                return {
                    "requests_per_minute": "60",
                    "requests_per_hour": "500",
                    "requests_per_day": "5000",
                }
            if "limits" in key:
                return {"max_total_tokens": "100000"}
            if "usage" in key:
                return {"prompt_tokens": "5000", "completion_tokens": "3000"}
            return {}

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side)
        mock_redis.zcount = AsyncMock(return_value=5)

        resp = billing_client.get("/billing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user-123"
        assert data["name"] == "Alice"
        assert len(data["usage"]) == 1
        assert data["usage"][0]["total_tokens"] == 8000
        assert len(data["limits"]) == 1
        assert data["limits"][0]["remaining_tokens"] == 92000
        assert data["rate_limit"]["requests_per_minute"] == 60
        assert data["rate_limit"]["requests_per_hour"] == 500
        assert data["rate_limit"]["requests_per_day"] == 5000
        assert data["rate_limit"]["current_requests_per_minute"] == 5
        assert data["rate_limit"]["current_requests_per_hour"] == 5
        assert data["rate_limit"]["current_requests_per_day"] == 5

    def test_billing_default_rate_limit(self, billing_client, mock_redis):
        """When no per-user rate config, uses DEFAULT_RATE_LIMIT."""
        mock_redis.scan = AsyncMock(side_effect=_mock_scan({}))
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_redis.zcount = AsyncMock(return_value=0)

        with patch("auth.DEFAULT_RATE_LIMIT", 120):
            resp = billing_client.get("/billing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rate_limit"]["requests_per_minute"] == 120
        assert data["rate_limit"]["requests_per_hour"] == 0
        assert data["rate_limit"]["requests_per_day"] == 0
        assert data["rate_limit"]["current_requests_per_minute"] == 0
        assert data["rate_limit"]["current_requests_per_hour"] == 0
        assert data["rate_limit"]["current_requests_per_day"] == 0
        assert data["usage"] == []
        assert data["limits"] == []
