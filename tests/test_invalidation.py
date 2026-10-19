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
async def test_cache_purge_by_model(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Sample output"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Populate entry for llama3.2:1b
        await ac.post(
            "/v1/chat/completions",
            json={"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Test prompt"}]},
        )

        # Populate entry for mistral
        await ac.post(
            "/v1/chat/completions",
            json={"model": "mistral", "messages": [{"role": "user", "content": "Another prompt"}]},
        )

        # Verify 2 keys stored
        keys_before = list(cache.vector_store.client.scan_iter("cache:llm:*"))
        assert len(keys_before) == 2

        # Purge only llama3.2:1b
        purge_resp = await ac.delete("/v1/cache/purge?model=llama3.2:1b")
        assert purge_resp.status_code == 200
        assert purge_resp.json()["purged_count"] == 1

        # Verify only mistral remains
        keys_after = list(cache.vector_store.client.scan_iter("cache:llm:*"))
        assert len(keys_after) == 1


@pytest.mark.asyncio
async def test_multi_turn_context_isolation(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload):
        nonlocal call_count
        call_count += 1
        return {
            "id": "chatcmpl-context",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Contextual answer"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Conversation A: Discussion on Python
        conv_a = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "user", "content": "Let's talk about programming."},
                {"role": "assistant", "content": "Sure, which language?"},
                {"role": "user", "content": "What is it?"},
            ],
            "temperature": 0.0,
        }

        # Conversation B: Discussion on Cooking with identical final prompt
        conv_b = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "user", "content": "Let's talk about baking bread."},
                {"role": "assistant", "content": "I love baking sourdough."},
                {"role": "user", "content": "What is it?"},
            ],
            "temperature": 0.0,
        }

        # Query A -> MISS (call 1)
        res_a = await ac.post("/v1/chat/completions", json=conv_a)
        assert res_a.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 1

        # Query B -> MUST MISS due to context hash difference, even with identical last prompt!
        res_b = await ac.post("/v1/chat/completions", json=conv_b)
        assert res_b.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 2