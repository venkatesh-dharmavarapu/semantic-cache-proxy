import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class ExactMatchCache:
    PREFIX = "cache:exact:"

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    @classmethod
    def generate_exact_key(
        cls,
        prompt: str,
        model: str,
        system_prompt_hash: str,
        params_hash: str,
        context_hash: str = "stateless",
        tenant_id: str = "default",
    ) -> str:
        normalized = f"{tenant_id}::{model.strip().lower()}::{system_prompt_hash}::{params_hash}::{context_hash}::{prompt.strip()}"
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{cls.PREFIX}{tenant_id}:{digest}"

    def lookup(self, exact_key: str) -> Optional[Tuple[Dict[str, Any], float, float]]:
        """
        Retrieves cached response along with XFetch metadata.
        Returns: (response_data, ttl_remaining, compute_delta) or None
        """
        client = self.vector_store.client
        if not client:
            return None
        try:
            raw = client.get(exact_key)
            if not raw:
                return None

            payload = json.loads(raw)
            meta = payload.get("_xfetch", {})
            logical_expires_at = meta.get("logical_expires_at", time.time() + 86400)
            compute_delta = meta.get("compute_delta", 0.5)
            ttl_remaining = max(0.0, logical_expires_at - time.time())

            response_data = payload.get("data", payload)
            return response_data, ttl_remaining, compute_delta
        except Exception as e:
            logger.warning("L1 exact cache lookup error: %s", e)
            return None

    def store_exact(
        self,
        exact_key: str,
        response_data: Dict[str, Any],
        ttl_seconds: int = 86400,
        compute_delta: float = 0.5,
        grace_period_seconds: int = 300,
    ):
        client = self.vector_store.client
        if not client:
            return
        try:
            envelope = {
                "data": response_data,
                "_xfetch": {
                    "logical_expires_at": time.time() + ttl_seconds,
                    "compute_delta": compute_delta,
                },
            }
            # Physical Redis TTL includes grace period for stale-while-revalidate
            physical_ttl = ttl_seconds + grace_period_seconds
            client.set(exact_key, json.dumps(envelope), ex=physical_ttl)
        except Exception as e:
            logger.warning("L1 exact cache store error: %s", e)

    store = store_exact