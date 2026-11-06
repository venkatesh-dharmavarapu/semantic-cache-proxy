import logging
import time
from typing import List, Optional
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class SessionThreadManager:
    SESSION_PREFIX = "session:"

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    def record_turn(
        self,
        session_id: str,
        turn_index: int,
        cache_key: str,
        ttl_seconds: int = 86400,
    ):
        """Records a session turn index mapped to its underlying cache key."""
        client = self.store.client
        if not client or not session_id:
            return

        session_key = f"{self.SESSION_PREFIX}{session_id}:turns"
        pipe = client.pipeline()
        pipe.zadd(session_key, {cache_key: turn_index})
        pipe.expire(session_key, ttl_seconds)
        pipe.execute()

    def prune_branches_after_turn(
        self,
        session_id: str,
        turn_index: int,
    ) -> int:
        """
        Invalidates and removes all cached turns occurring after turn_index (conversation rewinds/edits).
        Returns the number of invalidated turns.
        """
        client = self.store.client
        if not client or not session_id:
            return 0

        session_key = f"{self.SESSION_PREFIX}{session_id}:turns"
        # Retrieve all keys with turn_index greater than cutoff
        invalid_keys = client.zrangebyscore(session_key, min=turn_index + 1, max="+inf")
        if not invalid_keys:
            return 0

        pipe = client.pipeline()
        for k in invalid_keys:
            key_str = k.decode("utf-8") if isinstance(k, bytes) else k
            pipe.delete(key_str)
            # If it's an exact key or vector key, also remove references
            if key_str.startswith("cache:"):
                pipe.delete(f"feedback:entry:{key_str}")

        # Remove pruned entries from the sorted set
        pipe.zremrangebyscore(session_key, min=turn_index + 1, max="+inf")
        pipe.execute()

        logger.info(
            "Pruned %d downstream turns after turn %d for session %s",
            len(invalid_keys),
            turn_index,
            session_id,
        )
        return len(invalid_keys)

    def get_session_turns(self, session_id: str) -> List[tuple[str, int]]:
        """Returns ordered list of (cache_key, turn_index) for a session."""
        client = self.store.client
        if not client or not session_id:
            return []

        session_key = f"{self.SESSION_PREFIX}{session_id}:turns"
        entries = client.zrange(session_key, 0, -1, withscores=True)
        return [
            (k.decode("utf-8") if isinstance(k, bytes) else k, int(score))
            for k, score in entries
        ]