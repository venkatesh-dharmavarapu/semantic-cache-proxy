import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from redisvl.query import FilterQuery, VectorQuery
from app.config import settings
from app.core.cross_encoder import CrossEncoderVerifier
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
            num_results=max(top_k, 5),
        )

        try:
            dense_results = self.store.index.query(query)
            for d in dense_results:
                dense_id = d.get("id") or d.get("key")
                if dense_id and "vector_distance" in d:
                    dense_distance_map[dense_id] = float(d["vector_distance"])
        except Exception as e:
            logger.warning("Error querying vector index: %s", e)

        # 2. Query Sparse Text Index (BM25)
        sparse_results: List[Dict[str, Any]] = []
        if raw_prompt and self.store.index:
            try:
                sanitized_text = HybridSearchFusion.sanitize_text_query(raw_prompt)
                sparse_filter = f"{filter_expression} & (@prompt:({sanitized_text}))"
                sparse_query = FilterQuery(
                    return_fields=["prompt", "response_json", "hit_count"],
                    filter_expression=sparse_filter,
                    num_results=max(top_k, 5),
                )
                sparse_results = self.store.index.query(sparse_query)
            except Exception as e:
                logger.debug("Sparse query skipped or empty: %s", e)

        if not dense_results and not sparse_results:
            return None

        # Candidate selection with RRF
        candidate_record = None
        similarity_score = 0.0

        if dense_results and sparse_results:
            fused = HybridSearchFusion.compute_rrf_scores(dense_results, sparse_results)
            if fused:
                top_record, _ = fused[0]
                top_id = top_record.get("id") or top_record.get("key")
                vector_dist = dense_distance_map.get(
                    top_id, float(top_record.get("vector_distance", 0.0))
                )
                similarity_score = max(0.0, 1.0 - vector_dist)
                if similarity_score >= (threshold - 0.03):
                    candidate_record = top_record
        elif dense_results:
            top_result = dense_results[0]
            vector_distance = float(top_result.get("vector_distance", 1.0))
            similarity_score = max(0.0, 1.0 - vector_distance)
            if similarity_score >= threshold:
                candidate_record = top_result

        if not candidate_record:
            return None

        # 3. Stage 2 Cross-Encoder Intent Verification
        if raw_prompt:
            cached_prompt = candidate_record.get("prompt", "")
            is_valid, _ = CrossEncoderVerifier.verify_intent_consistency(
                query=raw_prompt,
                candidate_prompt=cached_prompt,
            )
            if not is_valid:
                logger.info(
                    "Cross-encoder rejected candidate '%s' for incoming query '%s'",
                    cached_prompt,
                    raw_prompt,
                )
                return None

        response_json = json.loads(candidate_record["response_json"])
        entry_key = candidate_record.get("id") or candidate_record.get("key")
        return response_json, similarity_score, entry_key

    def extend_key_ttl(self, entry_key: str, extension_seconds: int = 86400):
        try:
            if self.store.client:
                self.store.client.hincrby(entry_key, "hit_count", 1)
                self.store.client.expire(entry_key, extension_seconds)
        except Exception as e:
            logger.warning("Failed to extend TTL for %s: %s", entry_key, e)