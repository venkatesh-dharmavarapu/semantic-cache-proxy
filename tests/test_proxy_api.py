import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client


@pytest.fixture(autouse=True)
def clean_redis_keys():
    for key in cache.vector_store.client.scan_iter("cache:llm:*"):
        cache.vector_store.client.delete(key)
    yield
    for key in cache.vector_store.client.scan_iter("cache:llm:*"):
        cache.vector_store.client.delete(key)


@pytest.mark.asyncio
async def test_chat_completions_proxy_lifecycle(monkeypatch):
    # Mock 768-dimensional embedding vector (Ollama nomic-embed-text)
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    # Mock upstream Ollama response
    mock_upstream_response = {
        "id": "chatcmpl-proxy-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "llama3.2:1b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Proxied response content"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    call_count = 0

    async def mock_forward(payload):
        nonlocal call_count
        call_count += 1
        return mock_upstream_response

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "What is semantic caching?"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request 1 -> Must be MISS
        resp1 = await ac.post("/v1/chat/completions", json=req_body)
        assert resp1.status_code == 200
        assert resp1.headers["X-Cache-Lookup"] == "MISS"
        assert resp1.json()["choices"][0]["message"]["content"] == "Proxied response content"
        assert call_count == 1

        # Request 2 -> Must be HIT
        resp2 = await ac.post("/v1/chat/completions", json=req_body)
        assert resp2.status_code == 200
        assert resp2.headers["X-Cache-Lookup"] == "HIT"
        assert resp2.headers["X-Cache-Similarity"] != "0.0"
        assert resp2.json()["choices"][0]["message"]["content"] == "Proxied response content"
        # Upstream forwarder must NOT be called again
        assert call_count == 1