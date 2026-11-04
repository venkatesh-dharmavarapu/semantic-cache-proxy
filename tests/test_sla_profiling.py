import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.profiler import RequestProfiler
from app.core.sla_tracker import SLATracker


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


def test_profiler_timing_format():
    profiler = RequestProfiler()
    profiler.start("hash")
    # Simulate micro-work
    sum(range(1000))
    profiler.stop("hash")

    header = profiler.to_server_timing_header()
    assert "hash;dur=" in header


def test_sla_percentile_calculation():
    tracker = SLATracker(cache.vector_store)
    tracker.reset()

    # Populate 100 hit samples: 1.0 to 100.0 ms
    for i in range(1, 101):
        tracker.record_latency(is_hit=True, latency_ms=float(i))

    metrics = tracker.get_sla_metrics()
    hit_sla = metrics["cache_hits"]

    assert hit_sla["count"] == 100
    assert hit_sla["min"] == 1.0
    assert hit_sla["max"] == 100.0
    assert hit_sla["p50"] == 51.0
    assert hit_sla["p90"] == 91.0
    assert hit_sla["p99"] == 100.0


@pytest.mark.asyncio
async def test_server_timing_and_sla_api_endpoint(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload):
        return {
            "id": "chatcmpl-sla-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "SLA test response"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    req_body = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Test SLA latency"}],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request 1: MISS
        res1 = await ac.post("/v1/chat/completions", json=req_body)
        assert res1.status_code == 200
        assert "Server-Timing" in res1.headers
        assert "hash;dur=" in res1.headers["Server-Timing"]

        # Request 2: HIT (L1 fast-path)
        res2 = await ac.post("/v1/chat/completions", json=req_body)
        assert res2.status_code == 200
        assert res2.headers["X-Cache-Lookup"] == "HIT"
        assert "Server-Timing" in res2.headers
        assert "l1_exact;dur=" in res2.headers["Server-Timing"]

        # Query SLA endpoint
        sla_res = await ac.get("/v1/analytics/sla")
        assert sla_res.status_code == 200
        sla_data = sla_res.json()
        assert sla_data["cache_hits"]["count"] >= 1
        assert sla_data["cache_misses"]["count"] >= 1