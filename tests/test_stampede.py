import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.negative_cache import NegativeCachePolicy


@pytest.fixture(autouse=True)
def clean_redis_and_limits():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:llm:*"):
        client.delete(key)
    for key in client.scan_iter("lock:upstream:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:llm:*"):
        client.delete(key)
    for key in client.scan_iter("lock:upstream:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)


def test_negative_cache_policy_detection():
    clean_resp = {"choices": [{"message": {"content": "Here is how to sort a list."}}]}
    refusal_resp = {"choices": [{"message": {"content": "I am sorry, but I cannot assist with that."}}]}

    assert NegativeCachePolicy.is_negative_or_refusal(clean_resp) is False
    assert NegativeCachePolicy.is_negative_or_refusal(refusal_resp) is True
    assert NegativeCachePolicy.get_adaptive_ttl(refusal_resp, 86400) == 60
    assert NegativeCachePolicy.get_adaptive_ttl(clean_resp, 86400) == 86400


@pytest.mark.asyncio
async def test_anti_stampede_single_upstream_execution(monkeypatch):
    base_vector = [0.05] * 768

    async def mock_embed(text: str):
        return base_vector

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_slow_forward(payload):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.15)
        return {
            "id": "chatcmpl-stampede-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Stampede safe output"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_slow_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Heavy concurrent stampede query"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Fire 3 concurrent requests simultaneously
        tasks = [ac.post("/v1/chat/completions", json=req_body) for _ in range(3)]
        responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses)
        # Verify that upstream forwarder executed only once despite 3 concurrent calls
        assert call_count == 1