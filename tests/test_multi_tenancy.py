import math
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client, tenant_policy_manager
from app.core.tenant_policy import TenantConfig, TenantTier


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)
    for key in client.scan_iter("budget:*"):
        client.delete(key)

    # Force recreate index schema so tenant_id tag is active in Redis
    try:
        cache.vector_store.index.delete(drop=True)
    except Exception:
        pass
    cache.vector_store.create_index(overwrite=True)

    yield

    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)
    for key in client.scan_iter("budget:*"):
        client.delete(key)


@pytest.mark.asyncio
async def test_cross_tenant_cache_isolation(monkeypatch):
    """Verifies that identical prompts between two tenants do not bleed cache hits."""
    async def mock_embed(text: str):
        v = [0.0] * 768
        v[0] = 1.0
        return v

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-tenant-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Confidential Response {call_count}"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "What is our internal strategy roadmap?"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Tenant A executes query -> MISS -> caches response
        res_a1 = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_alpha"})
        assert res_a1.status_code == 200
        assert res_a1.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 1

        # Tenant A repeats exact query -> HIT
        res_a2 = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_alpha"})
        assert res_a2.status_code == 200
        assert res_a2.headers["X-Cache-Lookup"] == "HIT"
        assert call_count == 1

        # Tenant B sends the exact same prompt -> MUST BE A MISS (Zero cross-tenant leakage)
        res_b1 = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_beta"})
        assert res_b1.status_code == 200
        assert res_b1.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 2


@pytest.mark.asyncio
async def test_targeted_tenant_namespace_purge(monkeypatch):
    """Verifies that purging one tenant does not invalidate or affect other tenants."""
    async def mock_embed(text: str):
        v = [0.0] * 768
        v[0] = 1.0
        return v

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload):
        return {
            "id": "chatcmpl-purge-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Purge test response"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Sample tenant isolated query"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Populate cache for tenant_x and tenant_y
        await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_x"})
        await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_y"})

        # Confirm tenant_x has a cached entry
        rx_hit = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_x"})
        assert rx_hit.headers["X-Cache-Lookup"] == "HIT"

        # Execute targeted purge on tenant_x
        purge_res = await ac.delete("/v1/cache/tenants/tenant_x")
        assert purge_res.status_code == 200
        assert purge_res.json()["purged_keys_count"] >= 1

        # tenant_x must now MISS
        rx_after = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_x"})
        assert rx_after.headers["X-Cache-Lookup"] == "MISS"

        # tenant_y MUST STILL HIT (Uncompromised by tenant_x's purge)
        ry_hit = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_y"})
        assert ry_hit.headers["X-Cache-Lookup"] == "HIT"


@pytest.mark.asyncio
async def test_tier_policy_enforcement(monkeypatch):
    """Verifies that registered tenant tier configs enforce specific similarity thresholds."""
    # Register enterprise tenant with lower similarity threshold (0.80)
    tenant_policy_manager.register_tenant(
        "tenant_enterprise",
        TenantConfig(tier=TenantTier.ENTERPRISE, similarity_threshold=0.80),
    )

    async def mock_embed(text: str):
        if "how do i" in text.lower():
            # Seed query: unit vector along dim 0
            v = [0.0] * 768
            v[0] = 1.0
            return v
        else:
            # Paraphrased query: unit vector rotated such that cosine similarity ~= 0.848
            v = [0.0] * 768
            v[0] = 0.848
            v[1] = math.sqrt(1.0 - 0.848**2)
            return v

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload):
        return {
            "id": "chatcmpl-tier-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "List reversed"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Seed cache for enterprise tenant
        seed_req = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "How do I reverse a list in Python?"}],
            "temperature": 0.0,
        }
        await ac.post("/v1/chat/completions", json=seed_req, headers={"X-Tenant-ID": "tenant_enterprise"})

        # Paraphrased query retaining core intent terms so CrossEncoder passes
        paraphrase_req = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Reverse list in Python"}],
            "temperature": 0.0,
        }
        res = await ac.post(
            "/v1/chat/completions",
            json=paraphrase_req,
            headers={"X-Tenant-ID": "tenant_enterprise"},
        )
        assert res.status_code == 200
        assert res.headers["X-Cache-Lookup"] == "HIT"