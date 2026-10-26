import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client, rate_limiter


@pytest.fixture(autouse=True)
def clean_redis_and_limits():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:llm:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:llm:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)


@pytest.mark.asyncio
async def test_sliding_window_throttling_lifecycle(monkeypatch):
    # Save original settings
    original_limit = rate_limiter.default_limit
    original_window = rate_limiter.window_seconds

    # Temporarily lower thresholds for test
    rate_limiter.default_limit = 2
    rate_limiter.window_seconds = 10

    try:
        async def mock_embed(text: str):
            return [0.05] * 768

        monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

        async def mock_forward(payload):
            return {
                "id": "chatcmpl-ratelimit-test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "llama3.2:1b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Rate limit ok"},
                    }
                ],
            }

        monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

        req_body = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Test rate limits"}],
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Request 1 -> OK (remaining 1)
            r1 = await ac.post("/v1/chat/completions", json=req_body)
            assert r1.status_code == 200
            assert r1.headers["X-RateLimit-Remaining"] == "1"

            # Request 2 -> OK (remaining 0)
            r2 = await ac.post("/v1/chat/completions", json=req_body)
            assert r2.status_code == 200
            assert r2.headers["X-RateLimit-Remaining"] == "0"

            # Request 3 -> HTTP 429 Too Many Requests
            r3 = await ac.post("/v1/chat/completions", json=req_body)
            assert r3.status_code == 429
            assert "Retry-After" in r3.headers
            assert r3.json()["detail"]["error"]["type"] == "rate_limit_exceeded"
    finally:
        # Restore default rate limiter values
        rate_limiter.default_limit = original_limit
        rate_limiter.window_seconds = original_window