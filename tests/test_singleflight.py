import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.singleflight import SingleFlightCoordinator


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


@pytest.mark.asyncio
async def test_singleflight_coordinator_unit():
    coordinator = SingleFlightCoordinator()
    call_count = 0

    async def sample_work():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return {"data": "payload_ok"}, {"X-Test-Header": "unit"}

    # Launch 5 concurrent calls with same key
    tasks = [
        coordinator.execute_or_join("shared_key", sample_work)
        for _ in range(5)
    ]
    results = await asyncio.gather(*tasks)

    # Exactly 1 leader must execute
    assert call_count == 1

    leaders = [r for r in results if r[2] is True]
    followers = [r for r in results if r[2] is False]

    assert len(leaders) == 1
    assert len(followers) == 4

    for res, headers, _, _ in results:
        assert res["data"] == "payload_ok"
        assert "X-Test-Header" in headers


@pytest.mark.asyncio
async def test_proxy_coalescing_end_to_end(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    upstream_calls = 0

    async def mock_forward(payload, traceparent=None):
        nonlocal upstream_calls
        upstream_calls += 1
        await asyncio.sleep(0.1)  # Simulate upstream latency
        return {
            "id": f"chatcmpl-coalesce-{upstream_calls}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Coalesced response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "How does single-flight coalescing work?"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Fire 6 concurrent identical requests on an empty cache
        tasks = [
            ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "coalesce_tenant"})
            for _ in range(6)
        ]
        responses = await asyncio.gather(*tasks)

        # All 6 requests must succeed
        for r in responses:
            assert r.status_code == 200
            assert r.json()["choices"][0]["message"]["content"] == "Coalesced response"

        # Exactly 1 upstream call occurred
        assert upstream_calls == 1

        # Check that followers received coalesced headers
        coalesced_headers = [
            r.headers.get("X-Coalesced-Request") == "true"
            for r in responses
        ]
        assert coalesced_headers.count(True) == 5