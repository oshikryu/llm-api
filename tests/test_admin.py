"""Tests for admin.py endpoints."""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_redis():
    """Provide a mock Redis for admin tests."""
    mock_r = AsyncMock()
    mock_r.hgetall.return_value = {}
    mock_r.get.return_value = None
    mock_r.zcard.return_value = 0
    mock_r.zcount.return_value = 0
    mock_r.hincrby.return_value = 0
    mock_r.exists.return_value = True
    mock_r.smembers.return_value = set()
    mock_r.scan.return_value = ("0", [])
    from api import app
    app.state.redis = mock_r
    yield mock_r


@pytest.fixture()
def client(mock_redis):
    from api import app
    with (
        patch("api.http_client", AsyncMock()),
        patch("api.llm_semaphore", asyncio.Semaphore(4)),
    ):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


class TestCreateUser:
    def test_create_user(self, client, mock_redis):
        resp = client.post("/admin/users", json={"name": "Alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data
        mock_redis.hset.assert_called_once()

    def test_create_admin_user(self, client, mock_redis):
        resp = client.post("/admin/users", json={"name": "Bob", "is_admin": True})
        assert resp.status_code == 200
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]
        assert mapping["is_admin"] == "true"


class TestListUsers:
    def test_list_users(self, client, mock_redis):
        mock_redis.scan.return_value = ("0", ["llmapi:user:abc123"])
        mock_redis.hgetall.side_effect = [
            # First call: user data
            {"name": "Alice", "is_admin": "false", "created_at": "1700000000"},
            # Second call: rate limit config
            {"requests_per_minute": "60", "requests_per_hour": "500"},
        ]
        resp = client.get("/admin/users")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Alice"
        assert data[0]["rate_limits"]["requests_per_minute"] == 60
        assert data[0]["rate_limits"]["requests_per_hour"] == 500
        assert data[0]["rate_limits"]["requests_per_day"] == 0

    def test_list_users_empty(self, client, mock_redis):
        mock_redis.scan.return_value = ("0", [])
        resp = client.get("/admin/users")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetUser:
    def test_get_user(self, client, mock_redis):
        mock_redis.hgetall.side_effect = [
            {"name": "Alice", "is_admin": "false", "created_at": "1700000000"},
            {"requests_per_minute": "30", "requests_per_day": "1000"},
        ]
        resp = client.get("/admin/users/abc123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "abc123"
        assert data["rate_limits"]["requests_per_minute"] == 30
        assert data["rate_limits"]["requests_per_hour"] == 0
        assert data["rate_limits"]["requests_per_day"] == 1000

    def test_get_user_not_found(self, client, mock_redis):
        mock_redis.hgetall.return_value = {}
        resp = client.get("/admin/users/nonexistent")
        assert resp.status_code == 404


class TestDeleteUser:
    def test_delete_user(self, client, mock_redis):
        mock_redis.exists.return_value = True
        mock_redis.smembers.return_value = set()
        mock_redis.scan.return_value = ("0", [])
        resp = client.delete("/admin/users/abc123")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]

    def test_delete_user_not_found(self, client, mock_redis):
        mock_redis.exists.return_value = False
        resp = client.delete("/admin/users/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------


class TestCreateKey:
    def test_create_key(self, client, mock_redis):
        mock_redis.exists.return_value = True
        resp = client.post("/admin/users/abc123/keys")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"].startswith("llm-")
        assert len(data["key_prefix"]) == 12
        mock_redis.set.assert_called_once()
        mock_redis.sadd.assert_called_once()

    def test_create_key_user_not_found(self, client, mock_redis):
        mock_redis.exists.return_value = False
        resp = client.post("/admin/users/nonexistent/keys")
        assert resp.status_code == 404


class TestRevokeKey:
    def test_revoke_key(self, client, mock_redis):
        mock_redis.smembers.return_value = {"somehash:llm-abcdefgh"}
        resp = client.delete("/admin/users/abc123/keys/llm-abcdefgh")
        assert resp.status_code == 200
        mock_redis.delete.assert_called()
        mock_redis.srem.assert_called_once()

    def test_revoke_key_not_found(self, client, mock_redis):
        mock_redis.smembers.return_value = set()
        resp = client.delete("/admin/users/abc123/keys/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Limits & Rate Limit
# ---------------------------------------------------------------------------


class TestSetTokenLimit:
    def test_set_limit(self, client, mock_redis):
        mock_redis.exists.return_value = True
        resp = client.put(
            "/admin/users/abc123/limits/default",
            json={"max_total_tokens": 100000},
        )
        assert resp.status_code == 200
        mock_redis.hset.assert_called()

    def test_remove_limit(self, client, mock_redis):
        mock_redis.exists.return_value = True
        resp = client.put(
            "/admin/users/abc123/limits/default",
            json={"max_total_tokens": 0},
        )
        assert resp.status_code == 200
        mock_redis.delete.assert_called()

    def test_set_limit_user_not_found(self, client, mock_redis):
        mock_redis.exists.return_value = False
        resp = client.put(
            "/admin/users/nonexistent/limits/default",
            json={"max_total_tokens": 100},
        )
        assert resp.status_code == 404


class TestSetRateLimit:
    def test_set_rate_limit(self, client, mock_redis):
        mock_redis.exists.return_value = True
        resp = client.put(
            "/admin/users/abc123/rate-limit",
            json={"requests_per_minute": 120, "requests_per_hour": 500, "requests_per_day": 5000},
        )
        assert resp.status_code == 200
        # Verify all three fields stored
        mock_redis.hset.assert_any_call(
            "llmapi:ratelimit_config:abc123", "requests_per_minute", "120"
        )
        mock_redis.hset.assert_any_call(
            "llmapi:ratelimit_config:abc123", "requests_per_hour", "500"
        )
        mock_redis.hset.assert_any_call(
            "llmapi:ratelimit_config:abc123", "requests_per_day", "5000"
        )

    def test_set_rate_limit_backward_compat(self, client, mock_redis):
        """Old payload with only requests_per_minute still works."""
        mock_redis.exists.return_value = True
        resp = client.put(
            "/admin/users/abc123/rate-limit",
            json={"requests_per_minute": 60},
        )
        assert resp.status_code == 200
        mock_redis.hset.assert_called_once_with(
            "llmapi:ratelimit_config:abc123", "requests_per_minute", "60"
        )

    def test_set_rate_limit_zero_removes(self, client, mock_redis):
        """Setting hour/day to 0 removes the field (unlimited)."""
        mock_redis.exists.return_value = True
        resp = client.put(
            "/admin/users/abc123/rate-limit",
            json={"requests_per_hour": 0, "requests_per_day": 0},
        )
        assert resp.status_code == 200
        mock_redis.hdel.assert_any_call(
            "llmapi:ratelimit_config:abc123", "requests_per_hour"
        )
        mock_redis.hdel.assert_any_call(
            "llmapi:ratelimit_config:abc123", "requests_per_day"
        )

    def test_set_rate_limit_user_not_found(self, client, mock_redis):
        mock_redis.exists.return_value = False
        resp = client.put(
            "/admin/users/nonexistent/rate-limit",
            json={"requests_per_minute": 120},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class TestGetUsage:
    def test_get_usage(self, client, mock_redis):
        mock_redis.exists.return_value = True
        mock_redis.scan.return_value = ("0", ["llmapi:usage:abc123:default"])
        mock_redis.hgetall.return_value = {
            "prompt_tokens": "500",
            "completion_tokens": "300",
        }
        resp = client.get("/admin/users/abc123/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model"] == "default"
        assert data[0]["total_tokens"] == 800

    def test_get_usage_user_not_found(self, client, mock_redis):
        mock_redis.exists.return_value = False
        resp = client.get("/admin/users/nonexistent/usage")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Non-admin access
# ---------------------------------------------------------------------------


class TestNonAdminAccess:
    def test_non_admin_gets_403(self, mock_redis):
        """When auth is enabled, a non-admin user should get 403."""
        from auth import User, get_current_user
        from api import app

        non_admin = User(user_id="u1", name="Regular", is_admin=False)

        app.dependency_overrides[get_current_user] = lambda: non_admin
        try:
            with (
                patch("auth.AUTH_ENABLED", True),
                patch("api.http_client", AsyncMock()),
                patch("api.llm_semaphore", asyncio.Semaphore(4)),
            ):
                client = TestClient(app)
                resp = client.get("/admin/users")
                assert resp.status_code == 403
        finally:
            app.dependency_overrides.pop(get_current_user, None)
