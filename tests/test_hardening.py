import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache


@pytest.fixture(autouse=True)
def clean_redis_keys():
    for key in cache.vector_store.client.scan_iter("cache:llm:*"):
        cache.vector_store.client.delete(key)
    yield
    for key in cache.vector_store.client.scan_iter("cache:llm:*"):
        cache.vector_store.client.delete(key)


@pytest.mark.asyncio
async def test_semantic_cache_prewarming(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    warm_payload = {
        "items": [
            {
                "prompt": "What is Kubernetes?",
                "response_content": "Kubernetes is an open-source container orchestration system.",
                "model": "llama3.2:1b",
                "temperature": 0.0,
            },
            {
                "prompt": "What is Docker?",
                "response_content": "Docker is a containerization platform.",
                "model": "llama3.2:1b",
                "temperature": 0.0,
            },
        ]
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Pre-warm cache
        warm_resp = await ac.post("/v1/cache/warm", json=warm_payload)
        assert warm_resp.status_code == 200
        data = warm_resp.json()
        assert data["total_submitted"] == 2
        assert data["successfully_warmed"] == 2
        assert data["failures"] == 0

        # Query should now immediately HIT without contacting upstream
        query_payload = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "What is Kubernetes?"}],
            "temperature": 0.0,
        }
        hit_resp = await ac.post("/v1/chat/completions", json=query_payload)
        assert hit_resp.status_code == 200
        assert hit_resp.headers["X-Cache-Lookup"] == "HIT"
        content = hit_resp.json()["choices"][0]["message"]["content"]
        assert "container orchestration system" in content


@pytest.mark.asyncio
async def test_health_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Liveness check
        live_resp = await ac.get("/health/live")
        assert live_resp.status_code == 200
        assert live_resp.json()["status"] == "alive"

        # Readiness check (with running Redis)
        ready_resp = await ac.get("/health/ready")
        assert ready_resp.status_code in (200, 503)
        body = ready_resp.json()
        assert "dependencies" in body
        assert "redis" in body["dependencies"]