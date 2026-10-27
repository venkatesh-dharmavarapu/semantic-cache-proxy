import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.hasher import CacheKeyHasher
from app.core.quantization import VectorCompressor


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


def test_vector_compressor_math():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert pytest.approx(VectorCompressor.compute_cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(VectorCompressor.compute_cosine_similarity(v1, v3), 0.001) == 0.0

    b16 = VectorCompressor.to_float16_bytes([0.5, 0.25])
    assert len(b16) == 4  # 2 floats * 2 bytes


@pytest.mark.asyncio
async def test_near_duplicate_cluster_aliasing(monkeypatch):
    base_vector = [0.05] * 768

    async def mock_embed(text: str):
        return base_vector

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-cluster-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Cluster output"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    # Deterministic hashes matching standard parameters
    sys_hash = CacheKeyHasher.hash_system_prompt("")
    params_hash = CacheKeyHasher.hash_generation_params(model="llama3.2:1b", temperature=0.0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Prompt 1: Initial miss creates parent cluster
        r1 = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": "How to sort list?"}],
                "temperature": 0.0,
            },
        )
        assert r1.status_code == 200

        # Store near duplicate with matching partition hashes to test deduplication gate
        key = cache.store(
            prompt="Sort a list in Python",
            vector=base_vector,
            response_data={"choices": [{"message": {"content": "Cluster output"}}]},
            model="llama3.2:1b",
            system_prompt_hash=sys_hash,
            params_hash=params_hash,
            context_hash="stateless",
        )

        # Inspect cluster endpoint
        cluster_resp = await ac.get("/v1/cache/clusters")
        assert cluster_resp.status_code == 200
        stats = cluster_resp.json()
        assert stats["total_clusters"] == 1
        assert stats["consolidated_aliases"] >= 1