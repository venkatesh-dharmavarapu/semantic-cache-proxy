import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache
from app.core.drift_detector import SemanticDriftTracker


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


def test_drift_tracker_statistics():
    tracker = SemanticDriftTracker()
    tracker.reset()

    # Empty stats check
    empty_stats = tracker.get_distribution_stats()
    assert empty_stats["sample_count"] == 0

    # Record 10 hit similarity scores
    scores = [0.95, 0.94, 0.96, 0.92, 0.95, 0.93, 0.97, 0.95, 0.94, 0.95]
    for s in scores:
        tracker.record_hit_similarity(s)

    stats = tracker.get_distribution_stats()
    assert stats["sample_count"] == 10
    assert 0.94 <= stats["mean"] <= 0.96
    assert stats["drift_alert"] is False
    assert stats["p50"] > 0.90


@pytest.mark.asyncio
async def test_topology_and_tuning_endpoints(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    # Pre-populate sample cache cluster entries
    cache.store(
        prompt="Explain quicksort algorithm.",
        vector=[0.05] * 768,
        response_data={"choices": [{"message": {"content": "Quicksort is divide and conquer."}}]},
        model="llama3.2:1b",
        system_prompt_hash="sys_a",
        params_hash="params_a",
        context_hash="stateless",
    )

    tracker = SemanticDriftTracker()
    for s in [0.91, 0.92, 0.89, 0.90, 0.93, 0.91]:
        tracker.record_hit_similarity(s)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Check /v1/cache/drift
        drift_res = await ac.get("/v1/cache/drift")
        assert drift_res.status_code == 200
        d_body = drift_res.json()
        assert d_body["sample_count"] == 6

        # Check /v1/cache/topology
        topo_res = await ac.get("/v1/cache/topology")
        assert topo_res.status_code == 200
        t_body = topo_res.json()
        assert t_body["cluster_count"] >= 1
        assert "top_clusters" in t_body

        # Tune threshold dynamically
        tune_res = await ac.post("/v1/cache/threshold/tune", json={"target_confidence": 0.88})
        assert tune_res.status_code == 200
        tune_data = tune_res.json()
        assert tune_data["status"] == "updated"
        assert "new_threshold" in tune_data