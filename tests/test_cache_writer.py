import json
import pytest
from app.core.vector_store import VectorStoreManager
from app.core.cache_writer import CacheWriter


def test_cache_writer_payload():
    store = VectorStoreManager()
    store.connect()
    store.create_index()
    writer = CacheWriter(store)

    fake_vector = [0.05] * 768
    fake_response = {"id": "chat-123", "choices": [{"message": {"content": "Hello!"}}]}

    key = writer.store_response(
        prompt="Hi there",
        vector=fake_vector,
        response_data=fake_response,
        model="llama3.2:1b",
        system_prompt_hash="sys_default",
        params_hash="params_t0",
        ttl_seconds=300,
    )

    assert key.startswith("cache:llm:")

    stored_hash = store.client.hgetall(key)
    assert b"response_json" in stored_hash
    loaded_data = json.loads(stored_hash[b"response_json"].decode("utf-8"))
    assert loaded_data["choices"][0]["message"]["content"] == "Hello!"