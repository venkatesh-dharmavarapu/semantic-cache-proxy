import asyncio
import gzip
import json
import logging
import os
import time
from typing import Optional, Tuple
import numpy as np
from app.core.exact_cache import ExactMatchCache
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class CacheRehydrator:
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()

    async def rehydrate_from_file(
        self,
        filepath: str,
        batch_size: int = 250,
    ) -> Tuple[int, int]:
        """
        Rehydrates both L1 exact keys and L2 semantic vector indices from snapshot.
        Returns: (successfully_indexed, failed_records)
        """
        if not os.path.exists(filepath):
            logger.error("Snapshot file not found: %s", filepath)
            return 0, 0

        client = self.store.client
        if not client:
            return 0, 0

        exact_cache = ExactMatchCache(self.store)
        success_count = 0
        failure_count = 0
        batch: list[dict] = []

        def flush_batch(records: list[dict]):
            nonlocal success_count, failure_count
            if not records:
                return

            try:
                # 1. Bulk load into RedisVL SearchIndex
                self.store.index.load(records)
                # 2. Populate L1 exact match entries
                for rec in records:
                    resp_data = json.loads(rec["response_json"])
                    exact_key = ExactMatchCache.generate_exact_key(
                        prompt=rec["prompt"],
                        model=rec["model"],
                        system_prompt_hash=rec["system_prompt_hash"],
                        params_hash=rec["params_hash"],
                        context_hash=rec.get("context_hash", "stateless"),
                        tenant_id=rec.get("tenant_id", "default"),
                    )
                    exact_cache.store_exact(exact_key, resp_data, ttl_seconds=rec["ttl"])
                    success_count += 1
            except Exception as exc:
                logger.error("Failed to load rehydration batch: %s", exc)
                failure_count += len(records)

        with gzip.open(filepath, "rt", encoding="utf-8") as gz_in:
            for line in gz_in:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    vec_bytes = np.array(raw["prompt_vector"], dtype=np.float32).tobytes()

                    entry = {
                        "prompt": raw["prompt"],
                        "response_json": raw["response_json"],
                        "model": raw["model"],
                        "system_prompt_hash": raw["system_prompt_hash"],
                        "params_hash": raw["params_hash"],
                        "context_hash": raw.get("context_hash", "stateless"),
                        "tenant_id": raw.get("tenant_id", "default"),
                        "created_at": time.time(),
                        "ttl": raw.get("ttl", 86400),
                        "hit_count": raw.get("hit_count", 1),
                        "alias_count": 0,
                        "prompt_vector": vec_bytes,
                    }
                    batch.append(entry)

                    if len(batch) >= batch_size:
                        flush_batch(batch)
                        batch.clear()
                        await asyncio.sleep(0.01)  # Yield to event loop to keep proxy responsive
                except Exception as parse_err:
                    logger.warning("Error parsing snapshot line: %s", parse_err)
                    failure_count += 1

            if batch:
                flush_batch(batch)

        logger.info(
            "Snapshot rehydration complete: %d loaded, %d failed",
            success_count,
            failure_count,
        )
        return success_count, failure_count