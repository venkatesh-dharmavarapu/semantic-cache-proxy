import asyncio
import time
import uuid
from typing import Optional, Tuple
from app.core.vector_store import VectorStoreManager


class DistributedCacheLock:
    def __init__(self, vector_store: VectorStoreManager, lock_ttl_ms: int = 15000):
        self.store = vector_store
        self.lock_ttl_ms = lock_ttl_ms

    async def acquire_lock(
        self, lock_key: str, poll_interval: float = 0.05, timeout_s: float = 5.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Attempts to acquire a distributed lock.
        Returns:
            (is_leader: bool, lock_token: Optional[str])
            - (True, token): Primary worker responsible for executing upstream.
            - (False, None): Follower that waited while leader populated the cache.
        """
        token = str(uuid.uuid4())
        client = self.store.client
        if not client:
            return True, token

        full_key = f"lock:upstream:{lock_key}"

        # 1. Immediate acquisition attempt (The Leader)
        acquired = client.set(full_key, token, nx=True, px=self.lock_ttl_ms)
        if acquired:
            return True, token

        # 2. Sibling coroutines wait for the leader to complete and release the lock
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            # When the key no longer exists, the leader has finished and released
            if not client.exists(full_key):
                return False, None

        return False, None

    def release_lock(self, lock_key: str, token: Optional[str]):
        """Releases the distributed lock using a Lua script to ensure safe token ownership."""
        client = self.store.client
        if not client or not token:
            return

        lua_release = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        full_key = f"lock:upstream:{lock_key}"
        try:
            client.eval(lua_release, 1, full_key, token)
        except Exception:
            pass