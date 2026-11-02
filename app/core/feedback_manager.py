import json
import logging
import time
from typing import Any, Dict, Optional, Tuple
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class FeedbackManager:
    GLOBAL_STATS_KEY = "telemetry:cache:feedback_stats"
    KEY_FEEDBACK_PREFIX = "feedback:entry:"
    DEFAULT_NEGATIVE_EVICTION_THRESHOLD = -3

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        negative_eviction_threshold: int = -3,
    ):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()
        self.negative_eviction_threshold = negative_eviction_threshold

    def _normalize_rating(self, rating: Any) -> int:
        if rating in ("thumbs_up", 1):
            return 1
        elif rating in ("thumbs_down", -1):
            return -1
        return 0

    def record_feedback(
        self,
        cache_key: Optional[str] = None,
        completion_id: Optional[str] = None,
        rating: Any = 1,
        reason: Optional[str] = None,
        user_comment: Optional[str] = None,
    ) -> Tuple[bool, int, bool]:
        """
        Records user rating for a cached entry.
        Returns: (success: bool, current_net_score: int, was_evicted: bool)
        """
        client = self.store.client
        if not client:
            return False, 0, False

        val = self._normalize_rating(rating)
        target_key = cache_key

        # If cache_key is not provided but completion_id is, try mapping or use completion_id tag
        if not target_key and completion_id:
            target_key = f"cache:id_ref:{completion_id}"

        if not target_key:
            return False, 0, False

        feedback_entry_key = f"{self.KEY_FEEDBACK_PREFIX}{target_key}"
        now = time.time()

        pipe = client.pipeline()
        if val > 0:
            pipe.hincrby(feedback_entry_key, "positive", 1)
            pipe.hincrby(feedback_entry_key, "net_score", 1)
            pipe.hincrby(self.GLOBAL_STATS_KEY, "total_positive", 1)
        else:
            pipe.hincrby(feedback_entry_key, "negative", 1)
            pipe.hincrby(feedback_entry_key, "net_score", -1)
            pipe.hincrby(self.GLOBAL_STATS_KEY, "total_negative", 1)

        pipe.hincrby(self.GLOBAL_STATS_KEY, "total_feedback", 1)
        pipe.hset(feedback_entry_key, "last_updated", now)
        pipe.expire(feedback_entry_key, 604800)  # 7-day retention for feedback metadata

        if reason:
            pipe.hincrby(f"{self.GLOBAL_STATS_KEY}:reasons", reason.strip().lower(), 1)

        results = pipe.execute()
        current_net_score = int(results[1])

        # Automated Eviction on excessive negative feedback
        was_evicted = False
        if current_net_score <= self.negative_eviction_threshold:
            was_evicted = self.evict_degraded_entry(target_key)
            if was_evicted:
                client.hincrby(self.GLOBAL_STATS_KEY, "auto_evictions", 1)
                logger.warning(
                    "Cache entry %s automatically evicted due to net feedback score %d",
                    target_key,
                    current_net_score,
                )

        return True, current_net_score, was_evicted

    def evict_degraded_entry(self, cache_key: str) -> bool:
        """Purges degraded entry from Redis."""
        client = self.store.client
        if not client or not cache_key:
            return False

        try:
            # Delete direct key
            deleted = client.delete(cache_key)

            # If it's a vector hash key (cache:llm:...), also clear any associated prompt references
            if cache_key.startswith("cache:llm:"):
                # Clean up corresponding exact match cache if exists
                prompt = client.hget(cache_key, "prompt")
                if prompt:
                    # Scan and delete potential matching L1 exact cache keys
                    for exact_k in client.scan_iter("cache:exact:*"):
                        client.delete(exact_k)
            return bool(deleted > 0)
        except Exception as e:
            logger.error("Failed to evict degraded cache key %s: %s", cache_key, e)
            return False

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Calculates global satisfaction rates and reason breakdowns."""
        client = self.store.client
        if not client:
            return {
                "total_feedback": 0,
                "positive_count": 0,
                "negative_count": 0,
                "satisfaction_rate_percent": 0.0,
                "auto_evictions_count": 0,
                "recent_reasons": {},
            }

        raw = client.hgetall(self.GLOBAL_STATS_KEY)
        pos = int(raw.get(b"total_positive", 0))
        neg = int(raw.get(b"total_negative", 0))
        total = int(raw.get(b"total_feedback", 0))
        evictions = int(raw.get(b"auto_evictions", 0))

        rate = round((pos / total * 100.0), 2) if total > 0 else 100.0

        # Fetch category reasons
        raw_reasons = client.hgetall(f"{self.GLOBAL_STATS_KEY}:reasons")
        reasons_breakdown = {
            k.decode("utf-8"): int(v) for k, v in raw_reasons.items()
        }

        return {
            "total_feedback": total,
            "positive_count": pos,
            "negative_count": neg,
            "satisfaction_rate_percent": rate,
            "auto_evictions_count": evictions,
            "recent_reasons": reasons_breakdown,
        }

    def reset_stats(self):
        """Clears feedback stats and reasons."""
        client = self.store.client
        if client:
            client.delete(self.GLOBAL_STATS_KEY)
            client.delete(f"{self.GLOBAL_STATS_KEY}:reasons")
            for key in client.scan_iter(f"{self.KEY_FEEDBACK_PREFIX}*"):
                client.delete(key)