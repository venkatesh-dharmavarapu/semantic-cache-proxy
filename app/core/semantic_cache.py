import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.core.cache_reader import CacheReader
from app.core.cache_writer import CacheWriter
from app.core.embeddings import EmbeddingService
from app.core.hasher import CacheKeyHasher
from app.core.vector_store import VectorStoreManager


@dataclass
class CacheResult:
    is_hit: bool
    similarity_score: float
    response_data: Optional[Dict[str, Any]]
    cached_prompt: Optional[str]
    lookup_latency_ms: float
    cache_key: Optional[str]


class SemanticCache:
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: Optional[float] = None,
    ):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()
        self.store.create_index()

        self.embedder = embedding_service or EmbeddingService()
        self.reader = CacheReader(self.store)
        self.writer = CacheWriter(self.store)
        self.similarity_threshold = (
            similarity_threshold or settings.DEFAULT_SIMILARITY_THRESHOLD
        )

    def _extract_prompt_and_system(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, str]:
        """Separates system instructions from user/assistant conversation history."""
        system_prompts: List[str] = []
        conversation: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompts.append(content)
            else:
                conversation.append(f"{role}: {content}")

        system_text = "\n".join(system_prompts)
        user_query = conversation[-1] if conversation else ""
        return user_query, system_text

    async def lookup(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[CacheResult, List[float], str, str]:
        """
        Executes semantic lookup.
        Returns: (CacheResult, vector, system_prompt_hash, params_hash)
        """
        start_time = time.perf_counter()
        user_prompt, system_prompt = self._extract_prompt_and_system(messages)

        sys_hash = CacheKeyHasher.hash_system_prompt(system_prompt)
        p_hash = CacheKeyHasher.hash_generation_params(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

        threshold = similarity_threshold or self.similarity_threshold

        # Generate prompt embedding vector
        vector = await self.embedder.get_embedding(user_prompt)

        # Check nearest neighbor in Redis
        match = self.reader.query_similarity(
            vector=vector,
            model=model,
            system_prompt_hash=sys_hash,
            params_hash=p_hash,
            similarity_threshold=threshold,
        )

        lookup_time = (time.perf_counter() - start_time) * 1000.0

        if match:
            response_json, similarity, key = match
            return (
                CacheResult(
                    is_hit=True,
                    similarity_score=similarity,
                    response_data=response_json,
                    cached_prompt=None,
                    lookup_latency_ms=lookup_time,
                    cache_key=key,
                ),
                vector,
                sys_hash,
                p_hash,
            )

        return (
            CacheResult(
                is_hit=False,
                similarity_score=0.0,
                response_data=None,
                cached_prompt=None,
                lookup_latency_ms=lookup_time,
                cache_key=None,
            ),
            vector,
            sys_hash,
            p_hash,
        )

    def store(
        self,
        prompt: str,
        vector: List[float],
        response_data: Dict[str, Any],
        model: str,
        system_prompt_hash: str,
        params_hash: str,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Stores the miss result back into RedisVL index."""
        return self.writer.store_response(
            prompt=prompt,
            vector=vector,
            response_data=response_data,
            model=model,
            system_prompt_hash=system_prompt_hash,
            params_hash=params_hash,
            ttl_seconds=ttl_seconds,
        )