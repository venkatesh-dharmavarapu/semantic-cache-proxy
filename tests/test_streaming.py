import json
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
async def test_streaming_cache_lifecycle(monkeypatch):
    # Mock embedding
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    # Mock streaming generator from upstream Ollama
    async def mock_stream_chat(payload):
        chunks = ["Streaming ", "answers ", "from ", "local ", "Ollama."]
        for token in chunks:
            data_frame = {
                "id": "chatcmpl-stream-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2:1b",
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data_frame)}\n\n", None

        yield "data: [DONE]\n\n", None

        # Terminal emission with assembled payload
        full_payload = {
            "id": "chatcmpl-stream-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Streaming answers from local Ollama.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        yield "", full_payload

    monkeypatch.setattr(llm_client, "stream_chat_completion", mock_stream_chat)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Tell me a stream story."}],
        "stream": True,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request 1: Must be MISS with SSE streaming chunks
        resp1 = await ac.post("/v1/chat/completions", json=req_body)
        assert resp1.status_code == 200
        assert resp1.headers["X-Cache-Lookup"] == "MISS"
        assert "text/event-stream" in resp1.headers["content-type"]
        body1 = resp1.text
        assert "Streaming " in body1
        assert "data: [DONE]" in body1

        # Request 2: Must be HIT, replaying the assembled cached payload over SSE
        resp2 = await ac.post("/v1/chat/completions", json=req_body)
        assert resp2.status_code == 200
        assert resp2.headers["X-Cache-Lookup"] == "HIT"
        assert "text/event-stream" in resp2.headers["content-type"]
        body2 = resp2.text
        assert "Streaming" in body2
        assert "data: [DONE]" in body2