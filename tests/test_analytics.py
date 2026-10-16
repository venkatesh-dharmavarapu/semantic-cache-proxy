import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.core.telemetry import TelemetryTracker


@pytest.fixture(autouse=True)
def reset_telemetry():
    tracker = TelemetryTracker()
    tracker.reset_stats()
    yield
    tracker.reset_stats()


def test_telemetry_tracker_calculations():
    tracker = TelemetryTracker()

    # Record 3 hits, 1 miss, 1 bypass
    tracker.record_hit(latency_ms=15.0, upstream_est_latency_ms=800.0)
    tracker.record_hit(latency_ms=20.0, upstream_est_latency_ms=800.0)
    tracker.record_hit(latency_ms=25.0, upstream_est_latency_ms=800.0)
    tracker.record_miss(latency_ms=750.0)
    tracker.record_bypass()

    stats = tracker.get_stats()

    assert stats["total_requests"] == 5
    assert stats["hits"] == 3
    assert stats["misses"] == 1
    assert stats["bypasses"] == 1
    assert stats["hit_ratio_percent"] == 60.0
    assert stats["saved_latency_ms"] > 2000.0


@pytest.mark.asyncio
async def test_analytics_api_endpoints():
    tracker = TelemetryTracker()
    tracker.record_hit(latency_ms=10.0)
    tracker.record_miss(latency_ms=500.0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Check /v1/analytics/stats
        resp = await ac.get("/v1/analytics/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 2
        assert data["hits"] == 1
        assert data["misses"] == 1
        assert data["hit_ratio_percent"] == 50.0

        # Check /v1/analytics/reset
        reset_resp = await ac.post("/v1/analytics/reset")
        assert reset_resp.status_code == 200

        # Verify stats reset to zero
        stats_after = (await ac.get("/v1/analytics/stats")).json()
        assert stats_after["total_requests"] == 0
        assert stats_after["hits"] == 0