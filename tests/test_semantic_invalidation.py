import math
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    try:
        cache.vector_store.index.delete(drop=True)
    except Exception:
        pass
    cache.vector_store.create_index(overwrite=True)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


@pytest.mark.asyncio
async def test_semantic_range_sweep_and_dry_run(monkeypatch):
    # Vector simulation:
    # 'refund policy' queries point near unit vector [1.0, 0, ...]
    # 'shipping speed' query points orthogonal [0, 1.0, ...]
    async def mock_embed(text: str):
        v = [0.0] * 768
        t = text.lower()
        if "refund" in t or "return" in t:
            v[0] = 0.98
            v[1] = math.sqrt(1.0 - 0.98**2)
        elif "shipping" in t:
            v[2] = 1.0
        else:
            v[0] = 1.0
        return v

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload, traceparent=None):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-sweep-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Answer {call_count}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Seed two semantically related prompts and one distinct prompt
        req_refund1 = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "What is our customer refund policy?"}],
            "temperature": 0.0,
        }
        req_refund2 = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "How do returns and refunds work?"}],
            "temperature": 0.0,
        }
        req_shipping = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "What is the domestic shipping delivery speed?"}],
            "temperature": 0.0,
        }

        # Populate cache
        await ac.post("/v1/chat/completions", json=req_refund1, headers={"X-Tenant-ID": "tenant_sweep"})
        await ac.post("/v1/chat/completions", json=req_shipping, headers={"X-Tenant-ID": "tenant_sweep"})

        # Confirm both are cached
        r_ref_hit = await ac.post("/v1/chat/completions", json=req_refund1, headers={"X-Tenant-ID": "tenant_sweep"})
        assert r_ref_hit.headers["X-Cache-Lookup"] == "HIT"
        r_ship_hit = await ac.post("/v1/chat/completions", json=req_shipping, headers={"X-Tenant-ID": "tenant_sweep"})
        assert r_ship_hit.headers["X-Cache-Lookup"] == "HIT"

        # 2. Test Dry Run on "refund" seed prompt
        dry_res = await ac.post(
            "/v1/cache/invalidate/semantic",
            json={
                "prompt": "Our refund policy has been modified",
                "distance_threshold": 0.15,
                "tenant_id": "tenant_sweep",
                "dry_run": True,
            },
        )
        assert dry_res.status_code == 200
        dry_data = dry_res.json()
        assert dry_data["dry_run"] is True
        assert dry_data["evicted_count"] == 0
        assert dry_data["matched_count"] >= 1

        # Cache still hits after dry run
        r_after_dry = await ac.post("/v1/chat/completions", json=req_refund1, headers={"X-Tenant-ID": "tenant_sweep"})
        assert r_after_dry.headers["X-Cache-Lookup"] == "HIT"

        # 3. Execute Active Semantic Invalidation
        sweep_res = await ac.post(
            "/v1/cache/invalidate/semantic",
            json={
                "prompt": "Our refund policy has been modified",
                "distance_threshold": 0.15,
                "tenant_id": "tenant_sweep",
                "dry_run": False,
            },
        )
        assert sweep_res.status_code == 200
        sweep_data = sweep_res.json()
        assert sweep_data["dry_run"] is False
        assert sweep_data["evicted_count"] >= 1

        # 4. Refund query must now MISS (Evicted)
        r_ref_miss = await ac.post("/v1/chat/completions", json=req_refund1, headers={"X-Tenant-ID": "tenant_sweep"})
        assert r_ref_miss.headers["X-Cache-Lookup"] == "MISS"

        # 5. Shipping query MUST STILL HIT (Preserved outside blast radius)
        r_ship_still_hit = await ac.post("/v1/chat/completions", json=req_shipping, headers={"X-Tenant-ID": "tenant_sweep"})
        assert r_ship_still_hit.headers["X-Cache-Lookup"] == "HIT"