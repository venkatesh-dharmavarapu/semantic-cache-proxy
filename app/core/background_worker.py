import asyncio
import logging
from typing import Any, Dict, List
from app.core.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class BackgroundCacheWorker:
    @staticmethod
    def enqueue_cache_write(
        cache: SemanticCache,
        prompt: str,
        vector: List[float],
        response_data: Dict[str, Any],
        model: str,
        system_prompt_hash: str,
        params_hash: str,
        context_hash: str = "stateless",
    ):
        """
        Dispatches cache storage asynchronously in a non-blocking background task.
        Catches and logs any vector indexing exceptions without affecting client delivery.
        """
        async def _async_store():
            try:
                loop = asyncio.get_running_loop()
                # Run synchronous redisvl write in thread pool to avoid blocking async event loop
                await loop.run_in_executor(
                    None,
                    cache.store,
                    prompt,
                    vector,
                    response_data,
                    model,
                    system_prompt_hash,
                    params_hash,
                    context_hash,
                )
            except Exception as e:
                logger.error("Failed background cache write for prompt '%s': %s", prompt, e)

        asyncio.create_task(_async_store())