import asyncio
import os
import uuid

import pytest
from fastapi import HTTPException


pytestmark = pytest.mark.integration


def test_redis_stream_owner_and_rate_limit(monkeypatch):
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        pytest.skip("REDIS_URL is not configured")

    redis = pytest.importorskip("redis.asyncio")
    from backend.api import run_manager

    async def scenario():
        client = redis.from_url(redis_url, decode_responses=False)
        await client.flushdb()
        run_manager.set_redis_client(client)
        monkeypatch.setattr(run_manager, "_MAX_RUNS", 2)
        monkeypatch.setattr(run_manager, "_WINDOW_SECS", 60)
        try:
            run_id = f"itest-{uuid.uuid4().hex}"
            await run_manager.set_owner(run_id, "user-1")
            assert await run_manager.get_owner(run_id) == "user-1"

            await run_manager._publish_result(run_id, {"type": "step", "label": "ok"})
            event = await run_manager.read_result(run_id, "0-0")
            assert event is not None
            assert event["type"] == "step"
            assert event["_stream_id"]

            await run_manager.check_rate_limit("user-1")
            await run_manager.check_rate_limit("user-1")
            with pytest.raises(HTTPException) as exc:
                await run_manager.check_rate_limit("user-1")
            assert exc.value.status_code == 429
        finally:
            run_manager.set_redis_client(None)
            await client.aclose()

    asyncio.run(scenario())
