import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.tracer import W3CTraceContext


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)
    for key in client.scan_iter("budget:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("ratelimit:*"):
        client.delete(key)
    for key in client.scan_iter("budget:*"):
        client.delete(key)


def test_w3c_trace_context_parsing_and_child_generation():
    valid_parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    ctx = W3CTraceContext.from_header(valid_parent)
    assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert ctx.flags == "01"

    child = ctx.child_context()
    assert child.trace_id == ctx.trace_id
    assert child.parent_id == ctx.span_id
    assert child.span_id != ctx.span_id


@pytest.mark.asyncio
async def test_traceparent_propagation_and_metrics_accumulation(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    captured_headers = {}

    async def mock_forward(payload, traceparent=None):
        if traceparent:
            captured_headers["traceparent"] = traceparent
        return {
            "id": "chatcmpl-telemetry-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Telemetry test output"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    inbound_trace = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Test telemetry and traceparent"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request 1: MISS (Outbound call to LLM)
        res1 = await ac.post(
            "/v1/chat/completions",
            json=req_body,
            headers={"traceparent": inbound_trace, "X-Tenant-ID": "tenant_metrics"},
        )
        assert res1.status_code == 200
        assert "traceparent" in res1.headers
        assert res1.headers["traceparent"].startswith("00-4bf92f3577b34da6a3ce929d0e0e4736-")
        # Ensure trace was forwarded to upstream
        assert "traceparent" in captured_headers
        assert captured_headers["traceparent"].startswith("00-4bf92f3577b34da6a3ce929d0e0e4736-")

        # Request 2: HIT (L1 Exact)
        res2 = await ac.post(
            "/v1/chat/completions",
            json=req_body,
            headers={"X-Tenant-ID": "tenant_metrics"},
        )
        assert res2.status_code == 200
        assert res2.headers["X-Cache-Lookup"] == "HIT"

        # Query /metrics endpoint
        metrics_res = await ac.get("/metrics")
        assert metrics_res.status_code == 200
        metrics_text = metrics_res.text

        # Verify custom prometheus metrics exist in output
        assert "semantic_cache_requests_total" in metrics_text
        assert 'tier="L1-EXACT",status="HIT"' in metrics_text
        assert 'tier="NONE",status="MISS"' in metrics_text
        assert "semantic_cache_tokens_saved_total" in metrics_text
        assert "semantic_cache_cost_savings_usd_total" in metrics_text