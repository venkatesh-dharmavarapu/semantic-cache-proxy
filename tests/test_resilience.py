import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.exceptions import UpstreamTimeoutError


@pytest.fixture(autouse=True)
def clean_redis_keys():
    for key in cache.vector_store.client.scan_iter("cache:llm:*"):
        cache.vector_store.client.delete(key)
    yield
    for key in cache.vector_store.client.scan_iter("cache:llm:*"):
        cache.vector_store.client.delete(key)


@pytest.mark.asyncio
async def test_cache_bypass_header(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload):
        nonlocal call_count
        call_count += 1
        return {
            "id": "chatcmpl-bypass",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Bypass response",
                    },
                }
            ],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Test bypass query"}],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/v1/chat/completions",
            json=req_body,
            headers={"X-Cache-Bypass": "true"},
        )
        assert resp.status_code == 200
        assert resp.headers["X-Cache-Lookup"] == "BYPASS"
        assert call_count == 1


@pytest.mark.asyncio
async def test_upstream_timeout_error_handling(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    # Mock the LLM client call directly so the test harness HTTP client is unaffected
    async def mock_forward_timeout(payload):
        raise UpstreamTimeoutError("Ollama model execution timed out.")

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward_timeout)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Will timeout"}],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json=req_body)
        assert resp.status_code == 504
        assert resp.json()["detail"]["error"]["type"] == "upstream_timeout_error"