import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache
from app.core.feedback_manager import FeedbackManager


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("telemetry:*"):
        client.delete(key)
    for key in client.scan_iter("feedback:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("telemetry:*"):
        client.delete(key)
    for key in client.scan_iter("feedback:*"):
        client.delete(key)


def test_feedback_manager_lifecycle():
    manager = FeedbackManager(cache.vector_store, negative_eviction_threshold=-3)
    manager.reset_stats()

    # Pre-create a dummy key in Redis
    test_key = "cache:llm:dummy_test_key"
    cache.vector_store.client.hset(test_key, mapping={"prompt": "test prompt", "response_json": "{}"})

    # Record 2 positive ratings
    manager.record_feedback(cache_key=test_key, rating="thumbs_up")
    manager.record_feedback(cache_key=test_key, rating=1)

    # Check stats
    stats = manager.get_feedback_stats()
    assert stats["total_feedback"] == 2
    assert stats["positive_count"] == 2
    assert stats["satisfaction_rate_percent"] == 100.0

    # Record consecutive negative feedback until eviction threshold (-3)
    # Current net score: +2. We need 5 downvotes to reach net -3
    manager.record_feedback(cache_key=test_key, rating="thumbs_down", reason="hallucination")
    manager.record_feedback(cache_key=test_key, rating="thumbs_down", reason="hallucination")
    manager.record_feedback(cache_key=test_key, rating="thumbs_down", reason="outdated")
    manager.record_feedback(cache_key=test_key, rating="thumbs_down", reason="unhelpful")
    _, net_score, evicted = manager.record_feedback(cache_key=test_key, rating=-1, reason="unhelpful")

    assert net_score <= -3
    assert evicted is True

    # Verify key was deleted from Redis
    assert cache.vector_store.client.exists(test_key) == 0

    # Verify updated stats
    updated_stats = manager.get_feedback_stats()
    assert updated_stats["auto_evictions_count"] == 1
    assert "hallucination" in updated_stats["recent_reasons"]
    assert updated_stats["recent_reasons"]["hallucination"] == 2


@pytest.mark.asyncio
async def test_feedback_api_endpoints():
    test_key = "cache:llm:api_feedback_test"
    cache.vector_store.client.hset(test_key, mapping={"prompt": "hello", "response_json": "{}"})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Submit positive feedback via API
        req_payload = {
            "cache_key": test_key,
            "rating": "thumbs_up",
            "user_comment": "Very fast and accurate answer!",
        }
        res = await ac.post("/v1/feedback", json=req_payload)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["current_net_score"] == 1
        assert body["evicted"] is False

        # Check feedback stats endpoint
        stats_res = await ac.get("/v1/cache/feedback/stats")
        assert stats_res.status_code == 200
        stats_body = stats_res.json()
        assert stats_body["total_feedback"] == 1
        assert stats_body["positive_count"] == 1
        assert stats_body["satisfaction_rate_percent"] == 100.0