import logging
import uuid
from typing import Optional, Tuple
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class MutexLeaseManager:
    LEASE_PREFIX = "lease:recompute:"

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    def try_acquire_lease(
        self,
        key_identifier: str,
        lease_ttl_seconds: int = 15,
    ) -> Tuple[bool, Optional[str]]:
        """
        Attempts to acquire a non-blocking revalidation lease.
        Returns: (acquired: bool, lease_token: Optional[str])
        """
        client = self.store.client
        if not client:
            return False, None

        lease_key = f"{self.LEASE_PREFIX}{key_identifier}"
        lease_token = uuid.uuid4().hex

        # SET key value NX EX lease_ttl_seconds
        acquired = client.set(lease_key, lease_token, nx=True, ex=lease_ttl_seconds)
        if acquired:
            return True, lease_token
        return False, None

    def release_lease(self, key_identifier: str, lease_token: str) -> bool:
        """Releases the lease atomically using Lua script if token matches."""
        client = self.store.client
        if not client or not lease_token:
            return False

        lease_key = f"{self.LEASE_PREFIX}{key_identifier}"
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        try:
            res = client.eval(lua_script, 1, lease_key, lease_token)
            return bool(res)
        except Exception as exc:
            logger.error("Failed to release mutex lease for %s: %s", key_identifier, exc)
            return False