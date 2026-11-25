import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from redisvl.query import VectorRangeQuery
from app.core.exact_cache import ExactMatchCache
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class SemanticInvalidator:
    """
    Executes targeted semantic vector range sweeps to evict cached entries
    conceptually related to a seed prompt within an allowed blast radius.
    """

    MAX_BLAST_RADIUS_LIMIT = 200
    MAX_DISTANCE_CEILING = 0.25  # Equivalent to similarity >= 0.75

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    def find_entries_in_radius(
        self,
        seed_vector: List[float],
        distance_threshold: float = 0.12,
        tenant_id: Optional[str] = None,
        max_limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Executes a vector range query returning all entries within cosine distance.
        """
        if not self.store.index or not self.store.client:
            return []

        # Enforce safety ceiling on distance
        safe_distance = min(distance_threshold, self.MAX_DISTANCE_CEILING)
        safe_limit = min(max_limit, self.MAX_BLAST_RADIUS_LIMIT)

        filter_expr = f"(@tenant_id:{{{tenant_id}}})" if tenant_id else "*"

        # VectorRangeQuery in RedisVL finds vectors with distance <= distance_threshold
        query = VectorRangeQuery(
            vector=seed_vector,
            vector_field_name="prompt_vector",
            distance_threshold=safe_distance,
            filter_expression=filter_expr,
            num_results=safe_limit,
            return_fields=[
                "prompt",
                "model",
                "system_prompt_hash",
                "params_hash",
                "context_hash",
                "tenant_id",
                "schema_hash",
                "vector_distance",
            ],
        )

        try:
            results = self.store.index.query(query)
            return results or []
        except Exception as exc:
            logger.error("Failed to execute vector range query for invalidation: %s", exc)
            return []

    def evict_semantic_range(
        self,
        seed_vector: List[float],
        distance_threshold: float = 0.12,
        tenant_id: Optional[str] = None,
        max_limit: int = 50,
        dry_run: bool = False,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Identifies and removes matching vector hash keys and their associated L1 exact keys.
        Returns: (evicted_count, matched_records)
        """
        matched = self.find_entries_in_radius(
            seed_vector=seed_vector,
            distance_threshold=distance_threshold,
            tenant_id=tenant_id,
            max_limit=max_limit,
        )

        if dry_run or not matched:
            return 0, matched

        client = self.store.client
        if not client:
            return 0, matched

        pipe = client.pipeline()
        eviction_count = 0

        for record in matched:
            key = record.get("id") or record.get("key")
            if not key:
                continue

            # Delete L2 Vector record
            pipe.delete(key)
            eviction_count += 1

            # Delete corresponding L1 Exact Cache key
            exact_key = ExactMatchCache.generate_exact_key(
                prompt=record.get("prompt", ""),
                model=record.get("model", ""),
                system_prompt_hash=record.get("system_prompt_hash", "default"),
                params_hash=record.get("params_hash", "default"),
                context_hash=record.get("context_hash", "stateless"),
                tenant_id=record.get("tenant_id", "default"),
                schema_hash=record.get("schema_hash", "unstructured"),
            )
            pipe.delete(exact_key)

        pipe.execute()
        logger.info("Evicted %d entries via semantic blast radius sweep", eviction_count)
        return eviction_count, matched