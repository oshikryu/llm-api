"""Tests for auth.py — authentication, rate limiting, token tracking."""

import asyncio
import hashlib
import time

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_auth_disabled_returns_anonymous(self):
        from auth import get_current_user, ANONYMOUS_USER

        with patch("auth.AUTH_ENABLED", False):
            request = MagicMock()
            user = await get_current_user(request, redis=AsyncMock())
            assert user is ANONYMOUS_USER

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_401(self):
        from auth import get_current_user

        with patch("auth.AUTH_ENABLED", True):
            request = MagicMock()
            request.headers = {}
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(request, redis=AsyncMock())
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self):
        from auth import get_current_user

        with patch("auth.AUTH_ENABLED", True):
            request = MagicMock()
            request.headers = {"Authorization": "Bearer bad-key"}
            redis = AsyncMock()
            redis.get.return_value = None
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(request, redis=redis)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_bearer_key(self):
        from auth import get_current_user

        with patch("auth.AUTH_ENABLED", True):
            raw_key = "test-key-123"
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            request = MagicMock()
            request.headers = {"Authorization": f"Bearer {raw_key}"}
            redis = AsyncMock()
            redis.get.return_value = "user-42"
            redis.hgetall.return_value = {
                "name": "Test User",
                "is_admin": "false",
            }
            user = await get_current_user(request, redis=redis)
            assert user.user_id == "user-42"
            assert user.name == "Test User"
            assert user.is_admin is False
            redis.get.assert_called_once_with(f"llmapi:apikey:{key_hash}")

    @pytest.mark.asyncio
    async def test_valid_x_api_key_header(self):
        from auth import get_current_user

        with patch("auth.AUTH_ENABLED", True):
            raw_key = "my-api-key"
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            request = MagicMock()
            request.headers = {"X-API-Key": raw_key}
            redis = AsyncMock()
            redis.get.return_value = "user-99"
            redis.hgetall.return_value = {
                "name": "Alt User",
                "is_admin": "true",
            }
            user = await get_current_user(request, redis=redis)
            assert user.user_id == "user-99"
            assert user.is_admin is True


# ---------------------------------------------------------------------------
# check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    @pytest.mark.asyncio
    async def test_under_limit(self):
        from auth import check_rate_limit

        redis = AsyncMock()
        redis.hgetall.return_value = {"requests_per_minute": "60"}
        redis.zcount.return_value = 5
        # Should not raise
        await check_rate_limit("user-1", redis)
        redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_at_limit_raises_429(self):
        from auth import check_rate_limit

        redis = AsyncMock()
        redis.hgetall.return_value = {"requests_per_minute": "10"}
        redis.zcount.return_value = 10
        redis.zrangebyscore.return_value = []
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit("user-1", redis)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_over_limit_raises_429(self):
        from auth import check_rate_limit

        redis = AsyncMock()
        redis.hgetall.return_value = {"requests_per_minute": "5"}
        redis.zcount.return_value = 100
        redis.zrangebyscore.return_value = []
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit("user-1", redis)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_hour_limit_exceeded(self):
        from auth import check_rate_limit

        redis = AsyncMock()
        redis.hgetall.return_value = {
            "requests_per_minute": "60",
            "requests_per_hour": "100",
        }
        now_ms = int(time.time() * 1000)

        async def zcount_side(key, min_score, max_score):
            window_ms = now_ms - int(float(min_score))
            if window_ms < 100_000:  # ~minute window
                return 5  # under minute limit
            return 100  # at hour limit

        redis.zcount = AsyncMock(side_effect=zcount_side)
        redis.zrangebyscore.return_value = []
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit("user-1", redis)
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["limit_type"] == "requests_per_hour"

    @pytest.mark.asyncio
    async def test_day_limit_exceeded(self):
        from auth import check_rate_limit

        redis = AsyncMock()
        redis.hgetall.return_value = {
            "requests_per_minute": "60",
            "requests_per_hour": "500",
            "requests_per_day": "1000",
        }
        now_ms = int(time.time() * 1000)

        async def zcount_side(key, min_score, max_score):
            window_ms = now_ms - int(float(min_score))
            if window_ms < 100_000:  # ~minute window
                return 5
            if window_ms < 4_000_000:  # ~hour window
                return 50
            return 1000  # at day limit

        redis.zcount = AsyncMock(side_effect=zcount_side)
        redis.zrangebyscore.return_value = []
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit("user-1", redis)
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["limit_type"] == "requests_per_day"

    @pytest.mark.asyncio
    async def test_unlimited_tier_skipped(self):
        from auth import check_rate_limit

        redis = AsyncMock()
        # requests_per_hour = 0 means unlimited, should be skipped
        redis.hgetall.return_value = {
            "requests_per_minute": "60",
            "requests_per_hour": "0",
            "requests_per_day": "0",
        }
        redis.zcount.return_value = 5
        # Should not raise
        await check_rate_limit("user-1", redis)
        redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_after_in_429_detail(self):
        from auth import check_rate_limit

        redis = AsyncMock()
        redis.hgetall.return_value = {"requests_per_minute": "5"}
        redis.zcount.return_value = 5
        now_ms = int(time.time() * 1000)
        redis.zrangebyscore.return_value = [str(now_ms - 30_000)]  # oldest is 30s ago
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit("user-1", redis)
        detail = exc_info.value.detail
        assert detail["retry_after_seconds"] > 0
        assert detail["limit"] == 5
        assert detail["current"] == 5


# ---------------------------------------------------------------------------
# check_token_limit
# ---------------------------------------------------------------------------


class TestCheckTokenLimit:
    @pytest.mark.asyncio
    async def test_no_limit_set(self):
        from auth import check_token_limit

        redis = AsyncMock()
        redis.hgetall.return_value = {}
        # Should not raise (0 = unlimited)
        await check_token_limit("user-1", "default", redis)

    @pytest.mark.asyncio
    async def test_under_limit(self):
        from auth import check_token_limit

        redis = AsyncMock()
        redis.hgetall.side_effect = [
            {"max_total_tokens": "10000"},  # limits
            {"prompt_tokens": "1000", "completion_tokens": "2000"},  # usage
        ]
        await check_token_limit("user-1", "default", redis)

    @pytest.mark.asyncio
    async def test_at_limit_raises_403(self):
        from auth import check_token_limit

        redis = AsyncMock()
        redis.hgetall.side_effect = [
            {"max_total_tokens": "5000"},
            {"prompt_tokens": "3000", "completion_tokens": "2000"},
        ]
        with pytest.raises(HTTPException) as exc_info:
            await check_token_limit("user-1", "default", redis)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_over_limit_raises_403(self):
        from auth import check_token_limit

        redis = AsyncMock()
        redis.hgetall.side_effect = [
            {"max_total_tokens": "1000"},
            {"prompt_tokens": "800", "completion_tokens": "500"},
        ]
        with pytest.raises(HTTPException) as exc_info:
            await check_token_limit("user-1", "default", redis)
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# record_usage
# ---------------------------------------------------------------------------


class TestRecordUsage:
    @pytest.mark.asyncio
    async def test_increments_correctly(self):
        from auth import record_usage

        redis = AsyncMock()
        await record_usage("user-1", "default", 100, 50, redis)
        assert redis.hincrby.call_count == 2
        redis.hincrby.assert_any_call("llmapi:usage:user-1:default", "prompt_tokens", 100)
        redis.hincrby.assert_any_call("llmapi:usage:user-1:default", "completion_tokens", 50)

    @pytest.mark.asyncio
    async def test_skips_zero_values(self):
        from auth import record_usage

        redis = AsyncMock()
        await record_usage("user-1", "default", 0, 50, redis)
        assert redis.hincrby.call_count == 1
        redis.hincrby.assert_called_once_with("llmapi:usage:user-1:default", "completion_tokens", 50)

    @pytest.mark.asyncio
    async def test_skips_both_zero(self):
        from auth import record_usage

        redis = AsyncMock()
        await record_usage("user-1", "default", 0, 0, redis)
        redis.hincrby.assert_not_called()


# ---------------------------------------------------------------------------
# require_admin
# ---------------------------------------------------------------------------


class TestRequireAdmin:
    def test_admin_passes(self):
        from auth import require_admin, User

        require_admin(User(user_id="1", name="Admin", is_admin=True))

    def test_non_admin_raises_403(self):
        from auth import require_admin, User

        with pytest.raises(HTTPException) as exc_info:
            require_admin(User(user_id="1", name="User", is_admin=False))
        assert exc_info.value.status_code == 403
