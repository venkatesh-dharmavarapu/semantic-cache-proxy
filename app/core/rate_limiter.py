import time
from typing import Optional, Tuple
from app.config import settings
from app.core.vector_store import VectorStoreManager


class SlidingWindowRateLimiter:
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        default_limit: int = 60,
        window_seconds: int = 60,
    ):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()
        self.default_limit = default_limit
        self.window_seconds = window_seconds

    def check_rate_limit(
        self,
        client_identifier: str,
        limit: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """
        Evaluates sliding window limit for a client.
        Returns: (is_allowed: bool, remaining: int, retry_after_seconds: int)
        """
        max_requests = limit or self.default_limit
        window_size = window or self.window_seconds

        client = self.store.client
        if not client:
            return True, max_requests, 0

        now = time.time()
        window_start = now - window_size
        key = f"ratelimit:{client_identifier}"

        pipe = client.pipeline()
        # Remove timestamps outside current sliding window
        pipe.zremrangebyscore(key, "-inf", window_start)
        # Count requests inside current sliding window
        pipe.zcard(key)
        # Push current request timestamp
        pipe.zadd(key, {str(now): now})
        # Set TTL to ensure key cleanup
        pipe.expire(key, window_size + 1)
        results = pipe.execute()

        current_count = results[1]

        if current_count >= max_requests:
            # Over limit: pop the request just added
            client.zrem(key, str(now))
            # Calculate oldest event in window to determine retry cooldown
            oldest = client.zrange(key, 0, 0, withscores=True)
            retry_after = (
                int(window_size - (now - oldest[0][1]))
                if oldest
                else window_size
            )
            return False, 0, max(1, retry_after)

        remaining = max(0, max_requests - (current_count + 1))
        return True, remaining, 0