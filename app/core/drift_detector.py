import math
import time
from typing import Any, Dict, List, Optional
from app.core.vector_store import VectorStoreManager


class SemanticDriftTracker:
    HISTOGRAM_KEY = "telemetry:drift:similarities"
    MAX_SAMPLES = 1000

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    def record_hit_similarity(self, similarity_score: float):
        """Records a cache hit similarity score into the rolling evaluation window."""
        client = self.store.client
        if not client:
            return

        now = time.time()
        # Member format: "timestamp:score", score: similarity value
        member = f"{now}:{similarity_score:.4f}"

        pipe = client.pipeline()
        pipe.zadd(self.HISTOGRAM_KEY, {member: similarity_score})
        # Keep window bound to latest MAX_SAMPLES
        pipe.zremrangebyrank(self.HISTOGRAM_KEY, 0, -(self.MAX_SAMPLES + 1))
        pipe.execute()

    def get_distribution_stats(self) -> Dict[str, Any]:
        """Calculates mean, variance, standard deviation, and p50/p90/p99 similarity metrics."""
        client = self.store.client
        if not client:
            return {"sample_count": 0, "mean": 0.0, "std_dev": 0.0, "p50": 0.0, "p90": 0.0}

        raw_entries = client.zrange(self.HISTOGRAM_KEY, 0, -1, withscores=True)
        if not raw_entries:
            return {
                "sample_count": 0,
                "mean": 0.0,
                "std_dev": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "drift_alert": False,
            }

        scores = [score for _, score in raw_entries]
        n = len(scores)
        mean_val = sum(scores) / n
        variance = sum((x - mean_val) ** 2 for x in scores) / n if n > 1 else 0.0
        std_dev = math.sqrt(variance)

        # Percentile indexes
        p50 = scores[int(n * 0.50)]
        p90 = scores[int(n * 0.90)]

        # Alert if mean similarity drops close to edge boundary (< 0.82)
        drift_alert = mean_val < 0.82 and n >= 10

        return {
            "sample_count": n,
            "mean": round(mean_val, 4),
            "std_dev": round(std_dev, 4),
            "min": round(scores[0], 4),
            "max": round(scores[-1], 4),
            "p50": round(p50, 4),
            "p90": round(p90, 4),
            "drift_alert": drift_alert,
        }

    def reset(self):
        """Resets the similarity history."""
        if self.store.client:
            self.store.client.delete(self.HISTOGRAM_KEY)