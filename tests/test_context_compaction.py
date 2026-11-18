import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.context_compactor import ContextCompactor
from app.core.context_normalizer import ContextNormalizer


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


def test_conversational_filler_cleaning():
    raw = "  Hello, could you please explain Dijkstra's algorithm?   "
    cleaned = ContextCompactor.clean_text(raw)
    assert cleaned.lower().startswith("explain dijkstra's algorithm")
    assert not cleaned.lower().startswith("hello")


def test_small_context_bypasses_compaction():
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is Python?"},
    ]
    compacted, was_compacted, saved = ContextCompactor.compact_messages(messages, max_context_tokens=1000)
    assert was_compacted is False
    assert saved == 0
    assert len(compacted) == 3


def test_large_context_triggers_compaction():
    # Construct an oversized message history
    long_content = "This is a detailed code context explaining system architecture. " * 30
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": "Acknowledged initial architecture."},
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": "Acknowledged secondary architecture."},
        {"role": "user", "content": "How do I optimize the database queries?"},
    ]

    compacted, was_compacted, saved = ContextCompactor.compact_messages(
        messages, max_context_tokens=100, recent_anchor_turns=2
    )
    assert was_compacted is True
    assert saved > 0
    # System prompt preserved
    assert compacted[0]["role"] == "system"
    # Final user prompt preserved intact
    assert compacted[-1]["content"] == "How do I optimize the database queries?"


@pytest.mark.asyncio
async def test_proxy_compaction_headers(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    async def mock_forward(payload, traceparent=None):
        return {
            "id": "chatcmpl-compact-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2:1b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Compacted response"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    # Force normalizer threshold to a low token ceiling for testing
    from unittest.mock import patch
    with patch.object(ContextNormalizer, "partition_context") as mock_partition:
        mock_partition.return_value = ([], "stateless", "Prompt", True, 120)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            req = {
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": "Test prompt"}],
                "temperature": 0.0,
            }
            res = await ac.post("/v1/chat/completions", json=req)
            assert res.status_code == 200
            assert res.headers.get("X-Context-Compacted") == "true"
            assert res.headers.get("X-Tokens-Compacted") == "120"