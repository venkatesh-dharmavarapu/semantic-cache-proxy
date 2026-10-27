import json
import time
from typing import Any, Dict, List, Optional
import numpy as np
from app.config import settings
from app.core.quantization import VectorCompressor
from app.core.vector_store import VectorStoreManager


class CacheWriter:
    DEDUPLICATION_THRESHOLD = 0.96

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
        context_hash: str = "stateless",
        ttl_seconds: Optional[int] = None,
    ) -> str:
        ttl = ttl_seconds or settings.DEFAULT_CACHE_TTL_SECONDS
        now = time.time()
        vector_bytes = VectorCompressor.to_float32_bytes(vector)

        self.store.connect()

        # Check for near-identical duplicate to avoid cluster bloat
        if self.store.index and self.store.client:
            from app.core.cache_reader import CacheReader
            reader = CacheReader(self.store)
            near_duplicate = reader.query_similarity(
                vector=vector,
                model=model,
                system_prompt_hash=system_prompt_hash,
                params_hash=params_hash,
                context_hash=context_hash,
                similarity_threshold=self.DEDUPLICATION_THRESHOLD,
            )

            if near_duplicate:
                _, _, existing_key = near_duplicate
                # Update hit count and sliding TTL on the parent cluster key
                self.store.client.hincrby(existing_key, "alias_count", 1)
                self.store.client.expire(existing_key, ttl)
                return existing_key

        entry = {
            "prompt": prompt,
            "response_json": json.dumps(response_data),
            "model": model,
            "system_prompt_hash": system_prompt_hash,
            "params_hash": params_hash,
            "context_hash": context_hash,
            "created_at": now,
            "ttl": ttl,
            "hit_count": 0,
            "alias_count": 0,
            "prompt_vector": vector_bytes,
        }

        keys = self.store.index.load([entry])
        entry_key = keys[0]

        if self.store.client:
            self.store.client.expire(entry_key, ttl)

        return entry_key