import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, exact_cache, fallback_dispatcher, llm_client
from app.core.circuit_breaker import CircuitBreaker, CircuitState


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("circuit:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("circuit:*"):
        client.delete(key)


def test_circuit_breaker_state_transitions():
    cb = CircuitBreaker("test_provider", cache.vector_store, failure_threshold=2, recovery_timeout_seconds=0.5)
    cb.reset()

    assert cb.get_state() == CircuitState.CLOSED
    assert cb.can_execute() is True

    # Record 1 failure -> still closed
    cb.record_failure()
    assert cb.get_state() == CircuitState.CLOSED

    # Record 2nd failure -> trips to OPEN
    cb.record_failure()
    assert cb.get_state() == CircuitState.OPEN
    assert cb.can_execute() is False

    # Force reset
    cb.record_success()
    assert cb.get_state() == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_fallback_ladder_execution(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    attempted_models = []

    async def mock_forward(payload, traceparent=None):
        m = payload.get("model")
        attempted_models.append(m)
        if m == "llama3.2:1b":
            raise RuntimeError("Primary model timeout or 503 error")
        return {
            "id": "chatcmpl-fallback-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": m,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Fallback model response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Explain circuit breaker pattern"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/v1/chat/completions", json=req_body)
        assert res.status_code == 200
        # Check that primary was attempted first, then secondary
        assert attempted_models[0] == "llama3.2:1b"
        assert attempted_models[1] == "llama3.2:3b"
        assert res.headers.get("X-Upstream-Fallback") == "true"
        assert res.headers.get("X-Serving-Model") == "llama3.2:3b"


@pytest.mark.asyncio
async def test_stale_cache_salvage_on_complete_blackout(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward_fail(payload, traceparent=None):
        raise RuntimeError("Total upstream blackout")

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward_fail)

    # Seed an entry into the cache
    exact_key = exact_cache.generate_exact_key(
        prompt="Critical outage test",
        model="llama3.2:1b",
        system_prompt_hash=cache.reader.escape_redis_tag("empty") if hasattr(cache.reader, "escape_redis_tag") else "default",
        params_hash="default",
        context_hash="stateless",
        tenant_id="default",
    )
    exact_cache.store_exact(
        exact_key,
        {"choices": [{"message": {"role": "assistant", "content": "Salvaged emergency response"}}]},
        ttl_seconds=3600,
    )

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Critical outage test"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request should succeed with stale cache hit rather than returning a 503
        res = await ac.post("/v1/chat/completions", json=req_body)
        assert res.status_code == 200
        assert "HIT" in res.headers["X-Cache-Lookup"]