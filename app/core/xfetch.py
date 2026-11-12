import math
import random
import time
from typing import Optional


class XFetchEngine:
    """
    Implements the XFetch probabilistic early expiration algorithm:
    Ref: Vattani, Chierichetti, Lowenstein - "Optimal Probabilistic Cache Stampede Prevention"
    
    Formula:
        delta * beta * (-ln(rand())) > ttl_remaining
    """

    DEFAULT_BETA = 1.0
    DEFAULT_GRACE_PERIOD_SECONDS = 300.0

    @classmethod
    def should_refresh_early(
        cls,
        ttl_remaining: float,
        compute_delta_seconds: float,
        beta: float = DEFAULT_BETA,
    ) -> bool:
        """
        Determines whether the current reader should trigger an early background refresh.
        
        Args:
            ttl_remaining: Seconds remaining before logical expiration.
            compute_delta_seconds: Observed upstream compute time for this entry.
            beta: Aggressiveness multiplier (> 0). Higher beta refreshes earlier.
        """
        if ttl_remaining <= 0:
            return True

        if compute_delta_seconds <= 0:
            compute_delta_seconds = 0.5  # Sensible default latency baseline

        # Sample uniform distribution: (0, 1]
        u = random.random()
        while u == 0.0:
            u = random.random()

        # Asymmetric logarithmic hazard function
        hazard = compute_delta_seconds * beta * (-math.log(u))
        return hazard > ttl_remaining

    @staticmethod
    def pack_metadata(compute_duration_seconds: float, logical_ttl_seconds: int) -> dict:
        """Prepares metadata fields to be persisted alongside cache hashes."""
        now = time.time()
        return {
            "compute_delta": round(compute_duration_seconds, 4),
            "logical_expires_at": round(now + logical_ttl_seconds, 4),
            "stored_at": round(now, 4),
        }