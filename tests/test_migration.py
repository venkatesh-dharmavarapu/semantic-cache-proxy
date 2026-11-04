import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache


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


@pytest.mark.asyncio
async def test_batch_ingestion_and_query_hit(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    batch_payload = {
        "records": [
            {
                "prompt": "What is an abstract syntax tree?",
                "response": "An AST is a tree representation of the abstract syntactic structure of source code.",
                "model": "llama3.2:1b",
                "temperature": 0.0,
            },
            {
                "prompt": "Explain merge sort complexity.",
                "response": "Merge sort has a time complexity of O(n log n) in all cases.",
                "model": "llama3.2:1b",
                "temperature": 0.0,
            },
        ],
        "chunk_size": 2,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Ingest records
        ingest_res = await ac.post("/v1/cache/ingest/batch", json=batch_payload)
        assert ingest_res.status_code == 200
        res_data = ingest_res.json()
        assert res_data["status"] == "completed"
        assert res_data["successfully_indexed"] == 2
        assert res_data["failures"] == 0

        # Query AST prompt — should HIT without contacting upstream LLM
        query_body = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "What is an abstract syntax tree?"}],
            "temperature": 0.0,
        }
        hit_res = await ac.post("/v1/chat/completions", json=query_body)
        assert hit_res.status_code == 200
        assert hit_res.headers["X-Cache-Lookup"] == "HIT"
        content = hit_res.json()["choices"][0]["message"]["content"]
        assert "AST is a tree representation" in content


@pytest.mark.asyncio
async def test_schema_migration_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req = {"target_index_name": "llm_semantic_cache_v2"}
        migrate_res = await ac.post("/v1/cache/migrate", json=req)
        assert migrate_res.status_code == 200
        data = migrate_res.json()
        assert data["status"] == "ready"
        assert data["target_index"] == "llm_semantic_cache_v2"
        assert "total_keys_ready_for_cutover" in data