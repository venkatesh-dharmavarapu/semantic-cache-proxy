from typing import Dict, Any

# RedisVL index definition schema
INDEX_SCHEMA: Dict[str, Any] = {
    "index": {
        "name": "llm_semantic_cache",
        "prefix": "cache:llm:",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "prompt", "type": "text"},
        {"name": "response_json", "type": "text"},
        {"name": "model", "type": "tag"},
        {"name": "system_prompt_hash", "type": "tag"},
        {"name": "params_hash", "type": "tag"},
        {"name": "created_at", "type": "numeric"},
        {"name": "ttl", "type": "numeric"},
        {"name": "hit_count", "type": "numeric"},
        {
            "name": "prompt_vector",
            "type": "vector",
            "attrs": {
                "dims": 1536,
                "distance_metric": "cosine",
                "algorithm": "hnsw",
                "datatype": "float32",
                "m": 16,
                "ef_construction": 200,
                "ef_runtime": 10
            },
        },
    ],
}