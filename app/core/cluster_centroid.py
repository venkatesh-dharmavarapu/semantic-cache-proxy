import logging
import math
from typing import List, Optional, Tuple
import numpy as np
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class ClusterCentroidManager:
    """
    Manages semantic cluster centroids and aliases.
    Dynamically recalculates the representative embedding vector as a moving
    average of incoming queries matching the cluster to prevent centroid drift.
    """

    CLUSTER_PREFIX = "cluster:meta:"

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    @staticmethod
    def compute_moving_centroid(
        current_centroid: List[float],
        new_vector: List[float],
        current_count: int,
    ) -> List[float]:
        """
        Computes the incremental weighted average vector and normalizes to unit length (L2 norm).
        Formula: c_new = normalize( (c_old * N + v_new) / (N + 1) )
        """
        if current_count <= 0:
            return new_vector

        c_arr = np.array(current_centroid, dtype=np.float32)
        v_arr = np.array(new_vector, dtype=np.float32)

        # Weighted moving average
        updated = (c_arr * current_count + v_arr) / (current_count + 1)

        # Re-normalize to unit vector for cosine distance stability
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm

        return updated.tolist()

    def update_cluster_centroid(
        self,
        entry_key: str,
        incoming_vector: List[float],
        alias_prompt: str,
    ) -> Optional[List[float]]:
        """
        Updates the vector embedding and alias metadata for an existing cached entry.
        """
        client = self.store.client
        if not client:
            return None

        # Fetch current vector and alias count
        raw_vector = client.hget(entry_key, "prompt_vector")
        raw_alias_count = client.hget(entry_key, "alias_count")

        if not raw_vector:
            return None

        current_vector = np.frombuffer(raw_vector, dtype=np.float32).tolist()
        alias_count = int(raw_alias_count) if raw_alias_count else 0

        # Compute updated centroid
        new_centroid = self.compute_moving_centroid(
            current_centroid=current_vector,
            new_vector=incoming_vector,
            current_count=alias_count + 1,  # Base entry counts as 1
        )

        vector_bytes = np.array(new_centroid, dtype=np.float32).tobytes()

        pipe = client.pipeline()
        # Update vector and increment alias count
        pipe.hset(entry_key, "prompt_vector", vector_bytes)
        pipe.hincrby(entry_key, "alias_count", 1)

        # Track alias prompt in a set for administrative inspection
        alias_set_key = f"{self.CLUSTER_PREFIX}{entry_key}:aliases"
        pipe.sadd(alias_set_key, alias_prompt)
        pipe.expire(alias_set_key, 86400 * 7)  # 7-day retention
        pipe.execute()

        logger.debug(
            "Rebalanced centroid for '%s' (alias_count=%d)",
            entry_key,
            alias_count + 1,
        )
        return new_centroid

    def get_cluster_aliases(self, entry_key: str) -> List[str]:
        """Retrieves all query phrasings associated with a canonical cluster."""
        client = self.store.client
        if not client:
            return []

        alias_set_key = f"{self.CLUSTER_PREFIX}{entry_key}:aliases"
        members = client.smembers(alias_set_key)
        return [m.decode("utf-8") if isinstance(m, bytes) else m for m in members]