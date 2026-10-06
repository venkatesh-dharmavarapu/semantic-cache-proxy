import json
import time
from typing import Dict, Any, List
import numpy as np
from app.config import settings
from app.core.vector_store import VectorStoreManager

class CacheWriter:
    def __init__(self, vector_store: VectorStoreManager):
        self.store = vector_store

    def store_response(
        self,
        prompt: str,
        vector: List[float],
        response_data: Dict[str, Any],
        model: str,
        system_prompt_hash: str,
        params_hash: str,
        ttl_seconds: int | None = None,
    ) -> str:
        """
        Stores prompt embedding and metadata payload in Redis.
        Returns the created entry key.
        """
        ttl = ttl_seconds or settings.DEFAULT_CACHE_TTL_SECONDS
        now = time.time()
        
        # RedisVL expects raw float32 bytes for vector fields
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        entry = {
            "prompt": prompt,
            "response_json": json.dumps(response_data),
            "model": model,
            "system_prompt_hash": system_prompt_hash,
            "params_hash": params_hash,
            "created_at": now,
            "ttl": ttl,
            "hit_count": 0,
            "prompt_vector": vector_bytes,
        }
        
        # Load into Redis index
        keys = self.store.index.load([entry])
        entry_key = keys[0]
        
        # Set Redis native TTL key expiration as well
        self.store.client.expire(entry_key, ttl)
        return entry_key