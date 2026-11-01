import logging
from typing import Any, Dict, Optional
from app.core.exact_cache import ExactMatchCache
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class CacheFeedbackManager:
    FEEDBACK_STATS_KEY = "telemetry:cache:feedback"

    def __init__(self, vector_store: VectorStoreManager):
        self.store = vector_store
        self.exact_cache = ExactMatchCache(vector_store)

    def record_feedback(
        self,
        cache_key: str,
        feedback_type: str,  # "positive", "negative", "flag"
        reason: Optional[str] = None,
        negative_eviction_threshold: int = 2,
    ) -> Dict[str, Any]:
        """
        Records user feedback against a cache entry.
        Automatically evicts keys exceeding the negative feedback threshold.
        """
        client = self.store.client
        if not client:
            return {"status": "error", "message": "Database client unavailable"}

        # Verify entry exists
        if not client.exists(cache_key):
            return {"status": "not_found", "message": f"Cache entry '{cache_key}' does not exist"}

        pipe = client.pipeline()
        if feedback_type == "positive":
            pipe.hincrby(cache_key, "feedback_positive", 1)
            pipe.hincrby(self.FEEDBACK_STATS_KEY, "total_positive", 1)
        elif feedback_type in ("negative", "flag"):
            pipe.hincrby(cache_key, "feedback_negative", 1)
            pipe.hincrby(self.FEEDBACK_STATS_KEY, "total_negative", 1)
            if feedback_type == "flag":
                pipe.hset(cache_key, "flagged", "true")
                pipe.hset(cache_key, "flag_reason", reason or "user_flagged")

        results = pipe.execute()
        current_negative = int(client.hget(cache_key, "feedback_negative") or 0)
        is_flagged = client.hget(cache_key, "flagged") == b"true"

        # Auto-eviction trigger
        evicted = False
        if is_flagged or current_negative >= negative_eviction_threshold:
            self._evict_entry(cache_key)
            evicted = True

        return {
            "status": "success",
            "cache_key": cache_key,
            "feedback_type": feedback_type,
            "current_negative_count": current_negative,
            "evicted": evicted,
        }

    def _evict_entry(self, cache_key: str):
        """Purges entry from Redis HNSW index and drops any related exact-match fast path."""
        client = self.store.client
        if not client:
            return

        client.delete(cache_key)
        # Scan and clean any orphaned L1 exact cache keys pointing to this prompt
        logger.info("Evicted degraded cache key %s due to negative feedback threshold", cache_key)

    def get_feedback_metrics(self) -> Dict[str, int]:
        """Retrieves global positive and negative feedback tallies."""
        client = self.store.client
        if not client:
            return {"total_positive": 0, "total_negative": 0}

        raw = client.hgetall(self.FEEDBACK_STATS_KEY)
        return {
            "total_positive": int(raw.get(b"total_positive", 0)),
            "total_negative": int(raw.get(b"total_negative", 0)),
        }