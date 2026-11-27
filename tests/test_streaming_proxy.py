import json
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, exact_cache, llm_client
from app.core.streaming_pipe import StreamingTeePipe


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


@pytest.mark.asyncio
async def test_stream_cached_response_generator():
    cached_payload = {
        "id": "chatcmpl-test-stream",
        "model": "llama3.2:1b",
        "choices": [{"message": {"role": "assistant", "content": "Hello world from streaming cache!"}}],
    }

    chunks = []
    async for chunk in StreamingTeePipe.stream_cached_response(cached_payload, chunk_size_chars=5, delay_seconds=0):
        chunks.append(chunk)

    assert len(chunks) >= 3
    assert chunks[-1] == "data: [DONE]\n\n"

    # Verify first data chunk contains valid json
    first_data = chunks[0].replace("data: ", "").strip()
    parsed = json.loads(first_data)
    assert parsed["choices"][0]["delta"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_streaming_proxy_end_to_end(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    # Seed an entry into the cache
    exact_key = exact_cache.generate_exact_key(
        prompt="Stream test prompt",
        model="llama3.2:1b",
        system_prompt_hash=cache.reader.escape_redis_tag("empty") if hasattr(cache.reader, "escape_redis_tag") else "default",
        params_hash="default",
        context_hash="stateless",
        tenant_id="default",
        schema_hash="unstructured",
    )
    exact_cache.store_exact(
        exact_key,
        {"choices": [{"message": {"role": "assistant", "content": "Fast streaming hit"}}]},
        ttl_seconds=3600,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req_body = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Stream test prompt"}],
            "temperature": 0.0,
            "stream": True,
        }
        res = await ac.post("/v1/chat/completions", json=req_body)
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        assert res.headers["x-cache-lookup"] == "HIT"
        assert res.headers["x-cache-tier"] == "L1-EXACT"

        text = res.text
        assert "data: [DONE]" in text
        assert "Fast streaming hit" in text