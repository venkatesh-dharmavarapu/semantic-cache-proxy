import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from redisvl.query import VectorQuery
from app.config import settings
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class CacheReader:
    def __init__(self, vector_store: VectorStoreManager):
        self.store = vector_store

    def query_similarity(
        self,
        vector: List[float],
        model: str,
        system_prompt_hash: str,
        params_hash: str,
        context_hash: str = "stateless",
        similarity_threshold: Optional[float] = None,
        top_k: int = 1,
    ) -> Optional[Tuple[Dict[str, Any], float, str]]:
        """
        Queries Redis HNSW vector index for nearest semantic neighbors matching
        exact partition hashes.
        """
        threshold = similarity_threshold or settings.DEFAULT_SIMILARITY_THRESHOLD
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()

        # Build filter query for deterministic partitions
        filter_expression = (
            f"(@model:{{{model}}}) & "
            f"(@system_prompt_hash:{{{system_prompt_hash}}}) & "
            f"(@params_hash:{{{params_hash}}}) & "
            f"(@context_hash:{{{context_hash}}})"
        )

        query = VectorQuery(
            vector=vector_bytes,
            vector_field_name="prompt_vector",
            return_fields=["prompt", "response_json", "vector_distance", "hit_count"],
            filter_expression=filter_expression,
            num_results=top_k,
        )

        try:
            results = self.store.index.query(query)
            if not results:
                return None

            top_result = results[0]
            vector_distance = float(top_result.get("vector_distance", 1.0))
            similarity_score = 1.0 - vector_distance

            if similarity_score >= threshold:
                response_json = json.loads(top_result["response_json"])
                entry_key = top_result["id"]
                return response_json, similarity_score, entry_key

            return None
        except Exception as e:
            logger.warning("Error querying vector index: %s", e)
            return None

    def extend_key_ttl(self, entry_key: str, extension_seconds: int = 86400):
        """Applies adaptive sliding TTL extension upon cache hit."""
        try:
            if self.store.client:
                # Increment hit count and reset TTL window
                self.store.client.hincrby(entry_key, "hit_count", 1)
                self.store.client.expire(entry_key, extension_seconds)
        except Exception as e:
            logger.warning("Failed to extend TTL for %s: %s", entry_key, e)