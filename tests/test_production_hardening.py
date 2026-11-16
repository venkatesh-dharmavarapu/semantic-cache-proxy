import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.health import lifecycle_state
from app.api.proxy import cache, load_shedder


@pytest.fixture(autouse=True)
def reset_hardening_state():
    lifecycle_state.is_ready = True
    lifecycle_state.is_draining = False
    load_shedder.current_in_flight = 0
    yield
    lifecycle_state.is_ready = True
    lifecycle_state.is_draining = False
    load_shedder.current_in_flight = 0


@pytest.mark.asyncio
async def test_k8s_probes_lifecycle():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Liveness check
        live_res = await ac.get("/health/live")
        assert live_res.status_code == 200
        assert live_res.json()["status"] == "alive"

        # Readiness check (Connected to Redis)
        ready_res = await ac.get("/health/ready")
        assert ready_res.status_code == 200
        assert ready_res.json()["ready"] is True

        # Simulate draining transition
        lifecycle_state.is_draining = True
        draining_res = await ac.get("/health/ready")
        assert draining_res.status_code == 503
        assert draining_res.json()["ready"] is False


@pytest.mark.asyncio
async def test_load_shedder_rejection_under_saturation():
    # Force load shedder to artificial low limit
    load_shedder.max_concurrent_requests = 2
    load_shedder.current_in_flight = 2

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req_body = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Load shedding test"}],
            "temperature": 0.0,
        }
        res = await ac.post("/v1/chat/completions", json=req_body)
        assert res.status_code == 429
        assert "Retry-After" in res.headers
        assert "CONCURRENCY_SATURATED" in res.json()["detail"]


@pytest.mark.asyncio
async def test_request_rejection_during_draining():
    lifecycle_state.is_draining = True

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req_body = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Draining rejection test"}],
            "temperature": 0.0,
        }
        res = await ac.post("/v1/chat/completions", json=req_body)
        assert res.status_code == 503
        assert "shutting down" in res.json()["detail"].lower()