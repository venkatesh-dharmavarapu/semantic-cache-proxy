import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.context_normalizer import ContextNormalizer
from app.core.session_manager import SessionThreadManager


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("session:*"):
        client.delete(key)
    for key in client.scan_iter("telemetry:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    for key in client.scan_iter("session:*"):
        client.delete(key)
    for key in client.scan_iter("telemetry:*"):
        client.delete(key)


def test_context_normalizer_sliding_window():
    messages = [
        {"role": "system", "content": "You are a Python tutor."},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Answer 2"},
        {"role": "user", "content": "Turn 3"},
        {"role": "assistant", "content": "Answer 3"},
        {"role": "user", "content": "Current Prompt"},
    ]

    local_window, digest, prompt = ContextNormalizer.partition_context(messages, window_size=3)
    assert prompt == "Current Prompt"
    # Local window must contain the last 3 prior messages
    assert len(local_window) == 3
    assert local_window[-1]["content"] == "Answer 3"
    # Older turns must be hashed into digest
    assert digest != "stateless"
    assert digest != "root"


@pytest.mark.asyncio
async def test_multi_turn_session_caching_and_branch_pruning(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-turn-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Turn response {call_count}"}}],
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    session_id = "sess_multi_turn_99"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Turn 0
        req_turn0 = {
            "model": "llama3.2:1b",
            "messages": [{"role": "user", "content": "Hello, my project is called Aurora."}],
            "temperature": 0.0,
        }
        res0 = await ac.post(
            "/v1/chat/completions",
            json=req_turn0,
            headers={"X-Session-ID": session_id, "X-Turn-Index": "0"},
        )
        assert res0.status_code == 200
        assert res0.headers["X-Cache-Lookup"] == "MISS"

        # Turn 1
        req_turn1 = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "user", "content": "Hello, my project is called Aurora."},
                {"role": "assistant", "content": "Turn response 1"},
                {"role": "user", "content": "What is the project name?"},
            ],
            "temperature": 0.0,
        }
        res1 = await ac.post(
            "/v1/chat/completions",
            json=req_turn1,
            headers={"X-Session-ID": session_id, "X-Turn-Index": "1"},
        )
        assert res1.status_code == 200
        assert res1.headers["X-Cache-Lookup"] == "MISS"

        # Re-querying Turn 1 must HIT
        res1_hit = await ac.post(
            "/v1/chat/completions",
            json=req_turn1,
            headers={"X-Session-ID": session_id, "X-Turn-Index": "1"},
        )
        assert res1_hit.status_code == 200
        assert res1_hit.headers["X-Cache-Lookup"] == "HIT"

        # Prune conversation branches occurring after Turn 0 (rewinding turn 1)
        prune_res = await ac.post(
            "/v1/chat/sessions/prune",
            json={"session_id": session_id, "after_turn": 0},
        )
        assert prune_res.status_code == 200
        assert prune_res.json()["invalidated_turns_count"] >= 1

        # Querying Turn 1 now must MISS because downstream branch was invalidated
        res1_after_prune = await ac.post(
            "/v1/chat/completions",
            json=req_turn1,
            headers={"X-Session-ID": session_id, "X-Turn-Index": "1"},
        )
        assert res1_after_prune.status_code == 200
        assert res1_after_prune.headers["X-Cache-Lookup"] == "MISS"