import math
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.cluster_centroid import ClusterCentroidManager


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("cluster:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("cluster:*"):
        client.delete(key)


def test_moving_centroid_mathematics():
    v1 = [1.0, 0.0, 0.0] + [0.0] * 765
    v2 = [0.0, 1.0, 0.0] + [0.0] * 765

    # Combining v1 (count=1) with v2 -> equal weight average: [0.5, 0.5, 0] -> normalized: [1/sqrt(2), 1/sqrt(2), 0]
    c_new = ClusterCentroidManager.compute_moving_centroid(v1, v2, current_count=1)

    assert len(c_new) == 768
    expected_val = 1.0 / math.sqrt(2.0)
    assert pytest.approx(c_new[0], abs=1e-4) == expected_val
    assert pytest.approx(c_new[1], abs=1e-4) == expected_val

    # Verify unit length
    norm = np.linalg.norm(np.array(c_new, dtype=np.float32))
    assert pytest.approx(norm, abs=1e-5) == 1.0


@pytest.mark.asyncio
async def test_near_duplicate_aliasing_and_centroid_update(monkeypatch):
    # Vector generation that simulates high similarity (> 0.95)
    async def mock_embed(text: str):
        v = [0.0] * 768
        v[0] = 1.0
        if "invert" in text.lower():
            # Small perturbation maintaining ~0.96 cosine similarity
            v[0] = 0.96
            v[1] = math.sqrt(1.0 - 0.96**2)
        return v

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload, traceparent=None):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-rebalance-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "List reversed"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. First prompt seeds the cluster
        req1 = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Reverse a list in Python"}],
            "temperature": 0.0,
        }
        res1 = await ac.post("/v1/chat/completions", json=req1, headers={"X-Tenant-ID": "tenant_cluster"})
        assert res1.status_code == 200
        assert res1.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 1

        # 2. Second prompt is an alias with high semantic similarity
        req2 = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Invert a list in Python"}],
            "temperature": 0.0,
        }
        res2 = await ac.post("/v1/chat/completions", json=req2, headers={"X-Tenant-ID": "tenant_cluster"})
        assert res2.status_code == 200
        assert res2.headers["X-Cache-Lookup"] == "HIT"
        assert "L2" in res2.headers["X-Cache-Tier"]
        assert call_count == 1  # No upstream call

        # 3. Verify Redis only contains 1 canonical vector entry (no index bloat)
        client = cache.vector_store.client
        keys = list(client.scan_iter("cache:llm:*"))
        assert len(keys) == 1

        canonical_key = keys[0].decode("utf-8") if isinstance(keys[0], bytes) else keys[0]

        # 4. Check cluster inspection endpoint for registered alias
        alias_res = await ac.get(f"/v1/cache/clusters/{canonical_key}/aliases")
        assert alias_res.status_code == 200
        data = alias_res.json()
        assert data["alias_count"] >= 1
        assert "Invert a list in Python" in data["aliases"]