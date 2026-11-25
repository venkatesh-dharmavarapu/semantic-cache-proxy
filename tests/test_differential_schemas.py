import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.differential_engine import DifferentialEngine, SchemaHasher


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


def test_schema_hasher_isolation():
    schema1 = {"type": "json_schema", "json_schema": {"name": "User", "strict": True}}
    schema2 = {"type": "json_schema", "json_schema": {"name": "Order", "strict": True}}

    h_none = SchemaHasher.hash_response_format(None)
    h_s1 = SchemaHasher.hash_response_format(schema1)
    h_s2 = SchemaHasher.hash_response_format(schema2)

    assert h_none == "unstructured"
    assert h_s1 != "unstructured"
    assert h_s1 != h_s2


def test_json_patch_differential_math():
    base = {"name": "Alice", "role": "Engineer", "active": True}
    target = {"name": "Alice", "role": "Senior Engineer", "department": "Platform"}

    patch = DifferentialEngine.compute_patch(base, target)
    reconstructed = DifferentialEngine.apply_patch(base, patch)

    assert reconstructed == target
    assert reconstructed["role"] == "Senior Engineer"
    assert reconstructed.get("active") is None


@pytest.mark.asyncio
async def test_schema_mismatch_prevents_cache_hit(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload, traceparent=None):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-schema-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Schema payload {call_count}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    prompt = "Extract contact information from text"
    req_schema_a = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    req_schema_b = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "AddressModel"}},
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request 1: Schema A -> MISS
        res_a1 = await ac.post("/v1/chat/completions", json=req_schema_a)
        assert res_a1.status_code == 200
        assert res_a1.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 1
        hash_a = res_a1.headers["X-Schema-Hash"]

        # Request 2: Schema A again -> HIT
        res_a2 = await ac.post("/v1/chat/completions", json=req_schema_a)
        assert res_a2.status_code == 200
        assert res_a2.headers["X-Cache-Lookup"] == "HIT"
        assert call_count == 1

        # Request 3: Same prompt, but Schema B -> MUST MISS (No format collision)
        res_b = await ac.post("/v1/chat/completions", json=req_schema_b)
        assert res_b.status_code == 200
        assert res_b.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 2
        hash_b = res_b.headers["X-Schema-Hash"]
        assert hash_a != hash_b