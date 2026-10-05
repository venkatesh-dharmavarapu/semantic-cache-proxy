from app.core.hasher import CacheKeyHasher

def test_system_prompt_hashing():
    hash1 = CacheKeyHasher.hash_system_prompt("You are a helpful assistant.")
    hash2 = CacheKeyHasher.hash_system_prompt("  You are a helpful assistant. \n")
    assert hash1 == hash2

def test_param_hash_difference_on_temperature():
    hash_t0 = CacheKeyHasher.hash_generation_params(model="gpt-4o", temperature=0.0)
    hash_t1 = CacheKeyHasher.hash_generation_params(model="gpt-4o", temperature=0.7)
    assert hash_t0 != hash_t1