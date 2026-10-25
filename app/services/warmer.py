import asyncio
import logging
import time
from typing import Tuple
from app.core.hasher import CacheKeyHasher
from app.core.semantic_cache import SemanticCache
from app.schemas.warming_models import CacheWarmItem, CacheWarmRequest

logger = logging.getLogger(__name__)


class CacheWarmer:
    def __init__(self, cache: SemanticCache):
        self.cache = cache

    async def warm_item(self, item: CacheWarmItem) -> bool:
        """Embeds and indexes a single synthetic or historical item."""
        try:
            vector = await self.cache.embedder.get_embedding(item.prompt)
            sys_hash = CacheKeyHasher.hash_system_prompt(item.system_prompt or "")
            p_hash = CacheKeyHasher.hash_generation_params(
                model=item.model, temperature=item.temperature
            )

            canonical_response = {
                "id": f"chatcmpl-warmed-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": item.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": item.response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

            self.cache.store(
                prompt=item.prompt,
                vector=vector,
                response_data=canonical_response,
                model=item.model,
                system_prompt_hash=sys_hash,
                params_hash=p_hash,
                context_hash=item.context_hash or "stateless",
            )
            return True
        except Exception as e:
            logger.error("Error pre-warming item '%s': %s", item.prompt, e)
            return False

    async def warm_batch(self, request: CacheWarmRequest) -> Tuple[int, int]:
        """Concurrently warms a batch of items."""
        tasks = [self.warm_item(item) for item in request.items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        successes = sum(1 for r in results if r)
        failures = len(results) - successes
        return successes, failures