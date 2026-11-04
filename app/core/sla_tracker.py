import time
from typing import Any, Dict, List, Optional
from app.core.vector_store import VectorStoreManager


class SLATracker:
    HIT_LATENCIES_KEY = "telemetry:sla:latencies:hit"
    MISS_LATENCIES_KEY = "telemetry:sla:latencies:miss"
    MAX_SAMPLES = 2000

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    def record_latency(self, is_hit: bool, latency_ms: float):
        client = self.store.client
        if not client:
            return

        target_key = self.HIT_LATENCIES_KEY if is_hit else self.MISS_LATENCIES_KEY
        now = time.time()
        member = f"{now}:{latency_ms:.2f}"

        pipe = client.pipeline()
        pipe.zadd(target_key, {member: latency_ms})
        pipe.zremrangebyrank(target_key, 0, -(self.MAX_SAMPLES + 1))
        pipe.execute()

    def _calculate_percentiles(self, target_key: str) -> Dict[str, Any]:
        client = self.store.client
        if not client:
            return {"count": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

        entries = client.zrange(target_key, 0, -1, withscores=True)
        if not entries:
            return {"count": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

        scores = [score for _, score in entries]
        n = len(scores)

        def pct(p: float) -> float:
            idx = min(int(n * p), n - 1)
            return round(scores[idx], 2)

        return {
            "count": n,
            "min": round(scores[0], 2),
            "max": round(scores[-1], 2),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
        }

    def get_sla_metrics(self) -> Dict[str, Any]:
        return {
            "cache_hits": self._calculate_percentiles(self.HIT_LATENCIES_KEY),
            "cache_misses": self._calculate_percentiles(self.MISS_LATENCIES_KEY),
        }

    def reset(self):
        client = self.store.client
        if client:
            client.delete(self.HIT_LATENCIES_KEY)
            client.delete(self.MISS_LATENCIES_KEY)