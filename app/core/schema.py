from typing import Any, Dict
from app.config import settings

INDEX_SCHEMA: Dict[str, Any] = {
    "index": {
        "name": settings.CACHE_INDEX_NAME,
        "prefix": settings.CACHE_PREFIX,
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
                "dims": settings.EMBEDDING_DIM,
                "distance_metric": "cosine",
                "algorithm": "hnsw",
                "datatype": "float32",
                "m": 16,
                "ef_construction": 200,
                "ef_runtime": 10,
            },
        },
    ],
}