"""Tests for semaphore-based concurrency control."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from tests.conftest import make_completion_response


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Verify that _llm_post respects the semaphore limit."""
    import api

    call_count = 0
    max_concurrent = 0
    current_concurrent = 0

    original_post = AsyncMock()

    async def slow_post(*args, **kwargs):
        nonlocal call_count, max_concurrent, current_concurrent
        current_concurrent += 1
        max_concurrent = max(max_concurrent, current_concurrent)
        call_count += 1
        await asyncio.sleep(0.05)
        current_concurrent -= 1
        return make_completion_response()

    original_post.side_effect = slow_post

    with (
        patch.object(api, "http_client", original_post),
        patch.object(api, "llm_semaphore", asyncio.Semaphore(2)),
    ):
        # Override: http_client.post is what _llm_post calls
        original_post.post = AsyncMock(side_effect=slow_post)

        tasks = [api._llm_post("/completion", json={"prompt": "hi"}) for _ in range(6)]
        await asyncio.gather(*tasks)

    assert original_post.post.call_count == 6
    assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_semaphore_releases_on_error():
    """Verify semaphore is released even when the request raises."""
    import api

    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("connection error")

    sem = asyncio.Semaphore(2)

    with (
        patch.object(api, "http_client", mock_client),
        patch.object(api, "llm_semaphore", sem),
    ):
        with pytest.raises(Exception, match="connection error"):
            await api._llm_post("/completion", json={"prompt": "hi"})

    # Semaphore should be fully released
    assert sem._value == 2
