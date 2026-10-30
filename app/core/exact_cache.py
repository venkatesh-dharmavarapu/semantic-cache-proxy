import hashlib
import json
import logging
from typing import Any, Dict, Optional
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
    ) -> str:
        """Computes deterministic SHA-256 key for O(1) exact-match caching."""
        normalized = f"{model.strip().lower()}::{system_prompt_hash}::{params_hash}::{context_hash}::{prompt.strip()}"
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{cls.PREFIX}{digest}"

    def lookup(self, exact_key: str) -> Optional[Dict[str, Any]]:
        """O(1) exact match lookup bypassing vectorization entirely."""
        client = self.vector_store.client
        if not client:
            return None
        try:
            raw = client.get(exact_key)
            if raw:
                client.expire(exact_key, 86400)
                return json.loads(raw)
            return None
        except Exception as e:
            logger.warning("L1 exact cache lookup error: %s", e)
            return None

    def store_exact(
        self, exact_key: str, response_data: Dict[str, Any], ttl_seconds: int = 86400
    ):
        """Stores exact match payload in Redis without name collision with store attribute."""
        client = self.vector_store.client
        if not client:
            return
        try:
            client.set(exact_key, json.dumps(response_data), ex=ttl_seconds)
        except Exception as e:
            logger.warning("L1 exact cache store error: %s", e)

    # Alias for compatibility
    store = store_exact