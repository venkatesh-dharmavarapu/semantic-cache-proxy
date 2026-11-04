import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.hybrid_search import HybridSearchFusion


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


def test_rrf_scoring_logic():
    dense_list = [
        {"id": "doc1", "vector_distance": 0.05},
        {"id": "doc2", "vector_distance": 0.10},
    ]
    sparse_list = [
        {"id": "doc2", "score": 4.5},
        {"id": "doc3", "score": 3.2},
    ]

    fused = HybridSearchFusion.compute_rrf_scores(dense_list, sparse_list)
    assert len(fused) == 3

    # doc2 appears in both dense and sparse ranks -> receives combined boost
    top_doc, top_score = fused[0]
    assert top_doc["id"] == "doc2"
    assert top_score > fused[1][1]


@pytest.mark.asyncio
async def test_hybrid_search_end_to_end(monkeypatch):
    async def mock_embed(text: str):
        # Slightly noisy vector simulation
        val = 0.06 if "docker" in text.lower() else 0.05
        return [val] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload):
        return {
            "id": "chatcmpl-hybrid-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Docker multi-stage build guide."}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Explain Docker multi-stage builds"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Cold MISS
        res1 = await ac.post("/v1/chat/completions", json=req_body)
        assert res1.status_code == 200
        assert res1.headers["X-Cache-Lookup"] == "MISS"

        # 2. Similar query matching technical keywords -> HIT
        query_rephrase = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Docker multi-stage build explanation"}],
            "temperature": 0.0,
        }
        res2 = await ac.post("/v1/chat/completions", json=query_rephrase)
        assert res2.status_code == 200
        assert res2.headers["X-Cache-Lookup"] == "HIT"
        assert "Docker multi-stage build" in res2.json()["choices"][0]["message"]["content"]