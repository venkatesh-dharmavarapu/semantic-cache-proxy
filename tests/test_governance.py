import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.core.budget_guardrail import TenantBudgetGuardrail
from app.core.cost_calculator import TokenCostCalculator
from app.core.telemetry import TelemetryTracker


@pytest.fixture(autouse=True)
def clean_redis():
    tracker = TelemetryTracker()
    tracker.reset_stats()
    client = tracker.store.client
    for key in client.scan_iter("budget:tenant:*"):
        client.delete(key)
    yield
    tracker.reset_stats()
    for key in client.scan_iter("budget:tenant:*"):
        client.delete(key)


def test_token_cost_calculator():
    # 1000 prompt tokens + 500 completion tokens on llama3.2:1b
    # prompt: 1.0 * 0.00005 = 0.00005, completion: 0.5 * 0.00010 = 0.00005 -> Total = 0.00010
    cost = TokenCostCalculator.calculate_cost("llama3.2:1b", 1000, 500)
    assert cost == 0.0001

    payload = {
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
        }
    }
    p, c, t = TokenCostCalculator.extract_tokens(payload)
    assert p == 150
    assert c == 50
    assert t == 200


def test_tenant_budget_guardrail():
    guardrail = TenantBudgetGuardrail()
    tenant = "acme-corp"

    # Set quota of 500 tokens
    allowed, used = guardrail.check_quota(tenant, max_tokens=500)
    assert allowed is True
    assert used == 0

    # Consume 400 tokens
    current = guardrail.record_usage(tenant, 400)
    assert current == 400

    allowed, used = guardrail.check_quota(tenant, max_tokens=500)
    assert allowed is True
    assert used == 400

    # Consume 150 tokens (over quota)
    guardrail.record_usage(tenant, 150)
    allowed, used = guardrail.check_quota(tenant, max_tokens=500)
    assert allowed is False
    assert used == 550


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint():
    tracker = TelemetryTracker()
    tracker.record_hit(latency_ms=12.0)
    tracker.record_miss(latency_ms=650.0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        body = resp.text

        assert "semantic_cache_requests_total{status=\"hit\"} 1" in body
        assert "semantic_cache_requests_total{status=\"miss\"} 1" in body
        assert "semantic_cache_hit_ratio_percent 50.0" in body
        assert "semantic_cache_saved_latency_seconds_total" in body