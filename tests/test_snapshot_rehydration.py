import os
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, exact_cache, llm_client
from app.core.snapshot import SnapshotCheckpointer


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


@pytest.mark.asyncio
async def test_snapshot_export_and_rehydration_cycle(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload, traceparent=None):
        return {
            "id": "chatcmpl-snap-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Snapshot Answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "What is snapshot checkpointing?"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Populate cache
        res = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_snap"})
        assert res.status_code == 200
        assert res.headers["X-Cache-Lookup"] == "MISS"

        # Increment hit count so min_hits=1 picks it up
        res_hit = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_snap"})
        assert res_hit.headers["X-Cache-Lookup"] == "HIT"

        # 2. Export Snapshot
        export_res = await ac.post(
            "/v1/cache/snapshot/export",
            json={"min_hits": 1, "tenant_id": "tenant_snap", "filename": "test_export.jsonl.gz"},
        )
        assert export_res.status_code == 200
        data = export_res.json()
        filepath = data["snapshot_path"]
        assert data["exported_records"] >= 1
        assert os.path.exists(filepath)

        # 3. Simulate Total Disaster Recovery / Wipe Cache
        client = cache.vector_store.client
        for key in client.scan_iter("cache:*"):
            client.delete(key)

        # Verify query is now a cold MISS
        res_cold = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_snap"})
        # Temporarily mock forward to verify it would have hit upstream
        assert res_cold.headers["X-Cache-Lookup"] == "MISS"

        # Wipe again to test pure rehydration hit
        for key in client.scan_iter("cache:*"):
            client.delete(key)

        # 4. Rehydrate from Snapshot
        import_res = await ac.post("/v1/cache/snapshot/import", json={"filepath": filepath})
        assert import_res.status_code == 200
        assert import_res.json()["successfully_rehydrated"] >= 1

        # 5. Query immediately after rehydration -> MUST BE A HIT (Cold-start eliminated)
        res_rehydrated = await ac.post("/v1/chat/completions", json=req_body, headers={"X-Tenant-ID": "tenant_snap"})
        assert res_rehydrated.status_code == 200
        assert res_rehydrated.headers["X-Cache-Lookup"] == "HIT"

        # Clean up created file
        if os.path.exists(filepath):
            os.remove(filepath)