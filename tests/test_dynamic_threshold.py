import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.dynamic_threshold import DynamicThresholdCalibrator


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


def test_entropy_and_action_calibration():
    # Destructive action triggers higher threshold requirement
    calibrated_destructive, _, reason_dest = DynamicThresholdCalibrator.calibrate_threshold(
        prompt="drop table users", base_threshold=0.88
    )
    assert calibrated_destructive > 0.88
    assert "destructive_action" in reason_dest

    # Technical query with length relaxes threshold
    tech_prompt = "Explain Kubernetes pod topologySpreadConstraint scheduling with docker and pydantic"
    calibrated_tech, _, reason_tech = DynamicThresholdCalibrator.calibrate_threshold(
        prompt=tech_prompt, base_threshold=0.88
    )
    assert calibrated_tech < 0.88
    assert "high_token_count" in reason_tech or "technical_density" in reason_tech


@pytest.mark.asyncio
async def test_proxy_dynamic_threshold_headers(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload, traceparent=None):
        return {
            "id": "chatcmpl-thresh-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Calibration test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req_body = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Explain python asyncio event loop and fastapi"}],
            "temperature": 0.0,
        }
        res = await ac.post("/v1/chat/completions", json=req_body)
        assert res.status_code == 200
        assert "X-Adaptive-Threshold-Base" in res.headers
        assert "X-Adaptive-Threshold-Calibrated" in res.headers
        assert "X-Prompt-Entropy" in res.headers

        base = float(res.headers["X-Adaptive-Threshold-Base"])
        calibrated = float(res.headers["X-Adaptive-Threshold-Calibrated"])
        entropy = float(res.headers["X-Prompt-Entropy"])

        assert base > 0.0
        assert calibrated > 0.0
        assert entropy > 0.0