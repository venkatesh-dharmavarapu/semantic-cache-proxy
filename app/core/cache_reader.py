import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from redisvl.query import FilterQuery, VectorQuery
from app.config import settings
from app.core.hybrid_search import HybridSearchFusion
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def escape_redis_tag(val: str) -> str:
    """Escapes punctuation characters for RediSearch tag filters."""
    return re.sub(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\?])", r"\\\1", val)


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
        raw_prompt: Optional[str] = None,
    ) -> Optional[Tuple[Dict[str, Any], float, str]]:
        threshold = similarity_threshold or settings.DEFAULT_SIMILARITY_THRESHOLD
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()

        clean_model = escape_redis_tag(model)
        clean_sys = escape_redis_tag(system_prompt_hash)
        clean_params = escape_redis_tag(params_hash)
        clean_ctx = escape_redis_tag(context_hash)

        filter_expression = (
            f"(@model:{{{clean_model}}}) & "
            f"(@system_prompt_hash:{{{clean_sys}}}) & "
            f"(@params_hash:{{{clean_params}}}) & "
            f"(@context_hash:{{{clean_ctx}}})"
        )

        dense_results: List[Dict[str, Any]] = []
        dense_distance_map: Dict[str, float] = {}

        # 1. Query Dense Vector Index (HNSW)
        query = VectorQuery(
            vector=vector_bytes,
            vector_field_name="prompt_vector",
            return_fields=["prompt", "response_json", "vector_distance", "hit_count"],
            filter_expression=filter_expression,
            num_results=max(top_k, 3),
        )

        try:
            dense_results = self.store.index.query(query)
            for d in dense_results:
                dense_id = d.get("id") or d.get("key")
                if dense_id and "vector_distance" in d:
                    dense_distance_map[dense_id] = float(d["vector_distance"])
        except Exception as e:
            logger.warning("Error querying vector index: %s", e)

        # 2. Query Sparse Text Index (BM25) if raw prompt provided
        sparse_results: List[Dict[str, Any]] = []
        if raw_prompt and self.store.index:
            try:
                sanitized_text = HybridSearchFusion.sanitize_text_query(raw_prompt)
                sparse_filter = f"{filter_expression} & (@prompt:({sanitized_text}))"
                sparse_query = FilterQuery(
                    return_fields=["prompt", "response_json", "hit_count"],
                    filter_expression=sparse_filter,
                    num_results=max(top_k, 3),
                )
                sparse_results = self.store.index.query(sparse_query)
            except Exception as e:
                logger.debug("Sparse query skipped or empty: %s", e)

        # 3. Fuse via RRF if candidates present
        if not dense_results and not sparse_results:
            return None

        # If both dense and sparse returned results, fuse them
        if dense_results and sparse_results:
            fused = HybridSearchFusion.compute_rrf_scores(dense_results, sparse_results)
            if fused:
                top_record, _ = fused[0]
                top_id = top_record.get("id") or top_record.get("key")

                # Retrieve true vector distance from dense query if available
                if top_id in dense_distance_map:
                    vector_dist = dense_distance_map[top_id]
                else:
                    vector_dist = float(top_record.get("vector_distance", 0.0))

                similarity_score = max(0.0, 1.0 - vector_dist)
                if similarity_score >= (threshold - 0.03):
                    response_json = json.loads(top_record["response_json"])
                    return response_json, similarity_score, top_id

        # Fallback to top dense match
        if dense_results:
            top_result = dense_results[0]
            vector_distance = float(top_result.get("vector_distance", 1.0))
            similarity_score = max(0.0, 1.0 - vector_distance)

            if similarity_score >= threshold:
                response_json = json.loads(top_result["response_json"])
                entry_key = top_result.get("id") or top_result.get("key")
                return response_json, similarity_score, entry_key

        return None

    def extend_key_ttl(self, entry_key: str, extension_seconds: int = 86400):
        try:
            if self.store.client:
                self.store.client.hincrby(entry_key, "hit_count", 1)
                self.store.client.expire(entry_key, extension_seconds)
        except Exception as e:
            logger.warning("Failed to extend TTL for %s: %s", entry_key, e)