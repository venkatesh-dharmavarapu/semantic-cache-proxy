import pytest
from app.core.vector_store import VectorStoreManager
from app.core.cache_writer import CacheWriter
from app.core.cache_reader import CacheReader

def test_cache_hit_and_parameter_isolation():
    store = VectorStoreManager()
    store.connect()
    store.create_index()
    
    writer = CacheWriter(store)
    reader = CacheReader(store)
    
    vec = [0.1] * 1536
    resp = {"id": "test-response", "text": "cached output"}
    
    # Store with params_hash = 'param_a'
    writer.store_response(
        prompt="Explain Docker",
        vector=vec,
        response_data=resp,
        model="gpt-4o",
        system_prompt_hash="sys_v1",
        params_hash="param_a"
    )
    
    # 1. Matching query should HIT
    hit_result = reader.query_similarity(
        vector=vec,
        model="gpt-4o",
        system_prompt_hash="sys_v1",
        params_hash="param_a",
        similarity_threshold=0.90
    )
    assert hit_result is not None
    data, score, key = hit_result
    assert data["text"] == "cached output"
    assert score >= 0.99  # Identical vector
    
    # 2. Query with different params_hash should MISS even with identical vector
    miss_result = reader.query_similarity(
        vector=vec,
        model="gpt-4o",
        system_prompt_hash="sys_v1",
        params_hash="param_b",  # different param
        similarity_threshold=0.90
    )
    assert miss_result is None