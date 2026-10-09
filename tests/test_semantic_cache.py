import pytest
from unittest.mock import AsyncMock
from app.core.semantic_cache import SemanticCache
from app.core.vector_store import VectorStoreManager
from app.core.embeddings import EmbeddingService


@pytest.fixture
def mock_cache(monkeypatch):
    store = VectorStoreManager()
    store.connect()
    store.create_index()

    # Clean existing test entries to prevent state leakage between runs
    for key in store.client.scan_iter("cache:llm:*"):
        store.client.delete(key)

    embedder = EmbeddingService(api_key="sk-test")

    # Deterministic mock embedding based on query content
    async def mock_embed(text: str):
        val = 0.1 if "python" in text.lower() else 0.8
        return [val] * 1536

    monkeypatch.setattr(embedder, "get_embedding", mock_embed)
    cache = SemanticCache(
        vector_store=store, embedding_service=embedder, similarity_threshold=0.90
    )

    yield cache

    # Cleanup after test finishes
    for key in store.client.scan_iter("cache:llm:*"):
        store.client.delete(key)


@pytest.mark.asyncio
async def test_full_cache_lifecycle(mock_cache):
    messages = [
        {"role": "system", "content": "You are a code tutor."},
        {"role": "user", "content": "What is Python?"},
    ]
    model = "gpt-4o"
    temperature = 0.2

    # Step 1: Initial query must MISS
    res, vector, sys_hash, p_hash = await mock_cache.lookup(
        messages=messages, model=model, temperature=temperature
    )
    assert res.is_hit is False
    assert res.response_data is None

    # Step 2: Store mock response
    mock_llm_response = {
        "id": "chat-test-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Python is a high-level programming language.",
                }
            }
        ],
    }
    key = mock_cache.store(
        prompt="What is Python?",
        vector=vector,
        response_data=mock_llm_response,
        model=model,
        system_prompt_hash=sys_hash,
        params_hash=p_hash,
    )
    assert key.startswith("cache:llm:")

    # Step 3: Identical query must now HIT
    res2, _, _, _ = await mock_cache.lookup(
        messages=messages, model=model, temperature=temperature
    )
    assert res2.is_hit is True
    assert res2.similarity_score >= 0.99
    assert (
        res2.response_data["choices"][0]["message"]["content"]
        == "Python is a high-level programming language."
    )
    assert res2.lookup_latency_ms < 50.0

    # Step 4: Different temperature parameter must MISS
    res3, _, _, _ = await mock_cache.lookup(
        messages=messages, model=model, temperature=0.9
    )
    assert res3.is_hit is False