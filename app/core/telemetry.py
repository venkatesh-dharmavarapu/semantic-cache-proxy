import time
from typing import Any, Dict
from app.config import settings
from app.core.vector_store import VectorStoreManager


class TelemetryTracker:
    METRICS_KEY = "telemetry:cache:stats"

    def __init__(self, vector_store: VectorStoreManager | None = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    def record_hit(self, latency_ms: float, upstream_est_latency_ms: float = 800.0):
        """Record a cache hit and estimate time saved."""
        saved_ms = max(0.0, upstream_est_latency_ms - latency_ms)
        self.store.client.hincrby(self.METRICS_KEY, "total_requests", 1)
        self.store.client.hincrby(self.METRICS_KEY, "hits", 1)
        self.store.client.hincrbyfloat(self.METRICS_KEY, "saved_latency_ms", saved_ms)

    def record_miss(self, latency_ms: float):
        """Record a cache miss."""
        self.store.client.hincrby(self.METRICS_KEY, "total_requests", 1)
        self.store.client.hincrby(self.METRICS_KEY, "misses", 1)

    def record_bypass(self):
        """Record an explicit cache bypass."""
        self.store.client.hincrby(self.METRICS_KEY, "total_requests", 1)
        self.store.client.hincrby(self.METRICS_KEY, "bypasses", 1)

    def get_stats(self) -> Dict[str, Any]:
        """Fetch aggregated metrics and compute hit ratio."""
        raw = self.store.client.hgetall(self.METRICS_KEY)
        total = int(raw.get(b"total_requests", 0))
        hits = int(raw.get(b"hits", 0))
        misses = int(raw.get(b"misses", 0))
        bypasses = int(raw.get(b"bypasses", 0))
        saved_ms = float(raw.get(b"saved_latency_ms", 0.0))

        hit_ratio = round((hits / total * 100.0), 2) if total > 0 else 0.0

        return {
            "total_requests": total,
            "hits": hits,
            "misses": misses,
            "bypasses": bypasses,
            "hit_ratio_percent": hit_ratio,
            "saved_latency_ms": round(saved_ms, 2),
            "estimated_saved_seconds": round(saved_ms / 1000.0, 3),
        }

    def reset_stats(self):
        """Reset telemetry counters."""
        self.store.client.delete(self.METRICS_KEY)