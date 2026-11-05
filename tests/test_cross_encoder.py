import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.cross_encoder import CrossEncoderVerifier


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("telemetry:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("telemetry:*"):
        client.delete(key)


def test_cross_encoder_intent_rejection():
    # Contradicting operations on identical entity
    q1 = "How do I delete git branch test"
    c1 = "How do I create git branch test"
    is_valid, _ = CrossEncoderVerifier.verify_intent_consistency(q1, c1)
    assert is_valid is False

    # Paraphrase with matching intent
    q2 = "How to reverse list in Python"
    c2 = "Reverse a python list"
    is_valid, score = CrossEncoderVerifier.verify_intent_consistency(q2, c2)
    assert is_valid is True
    assert score >= 0.70


@pytest.mark.asyncio
async def test_cross_encoder_eliminates_contradictory_cache_hit(monkeypatch):
    # Simulate bi-encoder producing identical vectors for opposing operations
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-cross-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Operation response"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request 1: Ingest "delete branch" into cache
        req1 = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "How do I delete a Git branch?"}],
            "temperature": 0.0,
        }
        res1 = await ac.post("/v1/chat/completions", json=req1)
        assert res1.status_code == 200
        assert res1.headers["X-Cache-Lookup"] == "MISS"

        # Request 2: Opposing intent "create a Git branch"
        # Bi-encoder produces identical embedding, but Cross-Encoder must reject it -> MISS
        req2 = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "How do I create a Git branch?"}],
            "temperature": 0.0,
        }
        res2 = await ac.post("/v1/chat/completions", json=req2)
        assert res2.status_code == 200
        assert res2.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 2