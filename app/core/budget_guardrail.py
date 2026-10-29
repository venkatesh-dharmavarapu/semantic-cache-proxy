import time
from typing import Optional, Tuple
from app.core.vector_store import VectorStoreManager


class TenantBudgetGuardrail:
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    def record_usage(self, tenant_id: str, tokens_consumed: int, ttl_seconds: int = 86400) -> int:
        client = self.store.client
        if not client:
            return tokens_consumed
        key = f"budget:tenant:{tenant_id}:tokens"
        pipe = client.pipeline()
        pipe.incrby(key, tokens_consumed)
        pipe.expire(key, ttl_seconds)
        results = pipe.execute()
        return int(results[0])

    def check_quota(self, tenant_id: str, max_tokens: int) -> Tuple[bool, int]:
        client = self.store.client
        if not client:
            return True, 0
        key = f"budget:tenant:{tenant_id}:tokens"
        current_used = int(client.get(key) or 0)
        if current_used >= max_tokens:
            return False, current_used
        return True, current_used