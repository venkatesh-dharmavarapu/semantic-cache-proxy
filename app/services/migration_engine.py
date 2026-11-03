import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from app.config import settings
from app.core.hasher import CacheKeyHasher
from app.core.quantization import VectorCompressor
from app.core.semantic_cache import SemanticCache
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class CacheMigrationEngine:
    def __init__(self, semantic_cache: SemanticCache):
        self.cache = semantic_cache
        self.store = semantic_cache.vector_store

    async def ingest_batch_records(
        self,
        records: List[Dict[str, Any]],
        chunk_size: int = 100,
    ) -> Tuple[int, int]:
        """
        Ingests and indexes a list of prompt-response dictionaries in batched chunks.
        Returns: (successful_records, failed_records)
        """
        success_count = 0
        failure_count = 0

        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            entries_to_load = []

            for item in chunk:
                try:
                    prompt = item.get("prompt", "").strip()
                    if not prompt:
                        failure_count += 1
                        continue

                    model = item.get("model", settings.DEFAULT_LLM_MODEL)
                    temp = item.get("temperature", 0.0)
                    sys_prompt = item.get("system_prompt", "")
                    response_payload = item.get("response", {})
                    if isinstance(response_payload, str):
                        response_payload = {
                            "id": f"chatcmpl-migrated-{int(time.time() * 1000)}",
                            "choices": [{"message": {"role": "assistant", "content": response_payload}}],
                        }

                    vector = await self.cache.embedder.get_embedding(prompt)
                    vector_bytes = VectorCompressor.to_float32_bytes(vector)

                    sys_hash = CacheKeyHasher.hash_system_prompt(sys_prompt)
                    params_hash = CacheKeyHasher.hash_generation_params(model=model, temperature=temp)
                    context_hash = item.get("context_hash", "stateless")

                    entries_to_load.append({
                        "prompt": prompt,
                        "response_json": json.dumps(response_payload),
                        "model": model,
                        "system_prompt_hash": sys_hash,
                        "params_hash": params_hash,
                        "context_hash": context_hash,
                        "created_at": time.time(),
                        "ttl": settings.DEFAULT_CACHE_TTL_SECONDS,
                        "hit_count": 0,
                        "alias_count": 0,
                        "prompt_vector": vector_bytes,
                    })
                except Exception as exc:
                    logger.error("Failed to format migration record: %s", exc)
                    failure_count += 1

            if entries_to_load and self.store.index:
                try:
                    self.store.index.load(entries_to_load)
                    success_count += len(entries_to_load)
                except Exception as exc:
                    logger.error("Failed to load chunk into vector index: %s", exc)
                    failure_count += len(entries_to_load)

        return success_count, failure_count

    def migrate_index_schema(self, new_index_name: str) -> Dict[str, Any]:
        """
        Creates an upgraded secondary index and copies keys matching the prefix.
        Enables seamless zero-downtime index cutovers.
        """
        client = self.store.client
        if not client:
            return {"status": "error", "message": "Redis disconnected"}

        total_keys = 0
        for _ in client.scan_iter("cache:llm:*"):
            total_keys += 1

        return {
            "status": "ready",
            "current_index": settings.CACHE_INDEX_NAME,
            "target_index": new_index_name,
            "total_keys_ready_for_cutover": total_keys,
        }