import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, exact_cache, llm_client, mutex_lease
from app.core.xfetch import XFetchEngine


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("lease:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("lease:*"):
        client.delete(key)


def test_xfetch_decision_logic():
    # Immediate refresh on expired TTL
    assert XFetchEngine.should_refresh_early(ttl_remaining=0.0, compute_delta_seconds=1.0) is True
    assert XFetchEngine.should_refresh_early(ttl_remaining=-5.0, compute_delta_seconds=1.0) is True

    # High remaining TTL with low compute delta rarely refreshes
    refreshes = sum(
        XFetchEngine.should_refresh_early(ttl_remaining=10000.0, compute_delta_seconds=0.1)
        for _ in range(100)
    )
    assert refreshes == 0


def test_mutex_lease_acquisition_and_release():
    key_id = "test_prompt_hash_123"
    acquired, token = mutex_lease.try_acquire_lease(key_id, lease_ttl_seconds=10)
    assert acquired is True
    assert token is not None

    # Second acquisition attempt must fail
    acquired2, token2 = mutex_lease.try_acquire_lease(key_id, lease_ttl_seconds=10)
    assert acquired2 is False
    assert token2 is None

    # Release with invalid token fails
    assert mutex_lease.release_lease(key_id, "wrong_token") is False

    # Release with valid token succeeds
    assert mutex_lease.release_lease(key_id, token) is True

    # After release, acquisition succeeds again
    acquired3, token3 = mutex_lease.try_acquire_lease(key_id, lease_ttl_seconds=10)
    assert acquired3 is True
    mutex_lease.release_lease(key_id, token3)


@pytest.mark.asyncio
async def test_concurrent_reads_during_early_refresh_all_hit(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    upstream_calls = 0

    async def mock_forward(payload, traceparent=None):
        nonlocal upstream_calls
        upstream_calls += 1
        await asyncio.sleep(0.05)
        return {
            "id": f"chatcmpl-recompute-{upstream_calls}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Computed answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "What is the airspeed velocity of an unladen swallow?"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Prime cache
        res_init = await ac.post("/v1/chat/completions", json=req_body)
        assert res_init.status_code == 200
        assert res_init.headers["X-Cache-Lookup"] == "MISS"
        assert upstream_calls == 1

        # 2. Simulate nearing expiration by forcing XFetch to return True
        monkeypatch.setattr(XFetchEngine, "should_refresh_early", lambda *args, **kwargs: True)

        # 3. Fire 10 concurrent requests
        tasks = [ac.post("/v1/chat/completions", json=req_body) for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        # All 10 requests must immediately receive HIT without blocking
        for r in responses:
            assert r.status_code == 200
            assert r.headers["X-Cache-Lookup"] == "HIT"

        # Allow background recomputation task to finish
        await asyncio.sleep(0.2)

        # Exactly 1 background recomputation must have been scheduled
        assert upstream_calls == 2