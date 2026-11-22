import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client, request_hedger
from app.core.hedger import RequestHedger


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


@pytest.mark.asyncio
async def test_hedger_fast_primary_wins():
    hedger = RequestHedger(default_hedge_delay_ms=50.0)

    async def fast_primary():
        await asyncio.sleep(0.01)
        return {"result": "primary_fast"}

    secondary_called = False

    async def slow_secondary():
        nonlocal secondary_called
        secondary_called = True
        await asyncio.sleep(0.2)
        return {"result": "secondary_slow"}

    res, winner, was_hedged = await hedger.execute_hedged_race(
        primary_coro_fn=fast_primary,
        secondary_coro_fn=slow_secondary,
        hedge_delay_ms=50.0,
    )

    assert res["result"] == "primary_fast"
    assert winner == "primary"
    assert was_hedged is False
    assert secondary_called is False


@pytest.mark.asyncio
async def test_hedger_slow_primary_triggers_secondary_winner():
    hedger = RequestHedger(default_hedge_delay_ms=30.0)

    primary_cancelled = False

    async def slow_primary():
        nonlocal primary_cancelled
        try:
            await asyncio.sleep(0.5)
            return {"result": "primary_slow"}
        except asyncio.CancelledError:
            primary_cancelled = True
            raise

    async def fast_secondary():
        await asyncio.sleep(0.02)
        return {"result": "secondary_fast"}

    res, winner, was_hedged = await hedger.execute_hedged_race(
        primary_coro_fn=slow_primary,
        secondary_coro_fn=fast_secondary,
        hedge_delay_ms=20.0,
    )

    assert res["result"] == "secondary_fast"
    assert winner == "secondary"
    assert was_hedged is True
    # Trailing primary must be cancelled
    assert primary_cancelled is True


@pytest.mark.asyncio
async def test_proxy_hedging_integration(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_index = 0

    async def mock_forward(payload, traceparent=None):
        nonlocal call_index
        call_index += 1
        current_call = call_index

        if current_call == 1:
            # Primary call: simulate tail-latency stall (200ms)
            await asyncio.sleep(0.20)
            return {
                "id": "chatcmpl-primary",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "llama3.2:1b",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Primary slow answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        else:
            # Secondary hedge call: returns fast (10ms)
            await asyncio.sleep(0.01)
            return {
                "id": "chatcmpl-hedge-winner",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "llama3.2:1b",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hedged fast answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    # Set hedge delay to 40ms to trigger the speculative hedge
    monkeypatch.setattr(request_hedger, "default_hedge_delay_ms", 40.0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req_body = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Hedging end-to-end integration test"}],
            "temperature": 0.0,
        }
        res = await ac.post("/v1/chat/completions", json=req_body)
        assert res.status_code == 200
        assert res.headers.get("X-Hedging-Triggered") == "true"
        assert res.headers.get("X-Hedging-Winner") == "secondary"
        data = res.json()
        assert data["choices"][0]["message"]["content"] == "Hedged fast answer"