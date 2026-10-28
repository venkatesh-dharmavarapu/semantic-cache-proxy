import asyncio
import time
import uuid
from typing import Optional
from app.core.vector_store import VectorStoreManager


class DistributedCacheLock:
    def __init__(self, vector_store: VectorStoreManager, lock_ttl_ms: int = 15000):
        self.store = vector_store
        self.lock_ttl_ms = lock_ttl_ms

    async def acquire_lock(self, lock_key: str, poll_interval: float = 0.05, timeout_s: float = 5.0) -> Optional[str]:
        """
        Attempts to acquire a distributed lock.
        Returns lock token if acquired, or waits and returns None if released by another worker.
        """
        token = str(uuid.uuid4())
        client = self.store.client
        if not client:
            return token

        full_key = f"lock:upstream:{lock_key}"
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            acquired = client.set(full_key, token, nx=True, px=self.lock_ttl_ms)
            if acquired:
                return token
            await asyncio.sleep(poll_interval)

        return None

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