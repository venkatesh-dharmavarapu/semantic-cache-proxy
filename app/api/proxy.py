import time
from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import StreamingResponse
from app.core.distributed_lock import DistributedCacheLock
from app.core.exact_cache import ExactMatchCache
from app.core.exceptions import RateLimitExceededError
from app.core.hasher import CacheKeyHasher
from app.core.negative_cache import NegativeCachePolicy
from app.core.rate_limiter import SlidingWindowRateLimiter
from app.core.security_guardrail import CacheSecurityGuardrail
from app.core.semantic_cache import SemanticCache
from app.core.streaming import StreamingUtils
from app.core.telemetry import TelemetryTracker
from app.schemas.chat_models import ChatCompletionRequest
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/v1", tags=["Chat Proxy"])

cache = SemanticCache()
exact_cache = ExactMatchCache(cache.vector_store)
llm_client = LLMClient()
telemetry = TelemetryTracker()
rate_limiter = SlidingWindowRateLimiter(default_limit=60, window_seconds=60)
cache_lock = DistributedCacheLock(cache.vector_store)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    response: Response,
    x_cache_bypass: bool = Header(default=False),
    x_similarity_threshold: float | None = Header(default=None),
):
    client_ip = (
        req.headers.get("x-forwarded-for")
        or (req.client.host if req.client else "127.0.0.1")
    )

    # 1. Rate limiter check
    allowed, remaining, retry_after = rate_limiter.check_rate_limit(client_ip)
    if not allowed:
        raise RateLimitExceededError(
            retry_after=retry_after, limit=rate_limiter.default_limit
        )

    response.headers["X-RateLimit-Limit"] = str(rate_limiter.default_limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    serialized_messages = [msg.model_dump() for msg in request.messages]
    user_prompt, system_prompt = cache._extract_prompt_and_system(serialized_messages)

    sys_hash = CacheKeyHasher.hash_system_prompt(system_prompt)
    p_hash = CacheKeyHasher.hash_generation_params(
        model=request.model, temperature=request.temperature, max_tokens=request.max_tokens
    )
    c_hash = CacheKeyHasher.hash_conversation_context(serialized_messages)

    # 2. Handle Explicit Cache Bypass
    if x_cache_bypass:
        telemetry.record_bypass()
        payload = request.model_dump(exclude_none=True)

        if request.stream:
            async def bypass_stream_generator():
                async for chunk_str, _ in llm_client.stream_chat_completion(payload):
                    if chunk_str:
                        yield chunk_str

            return StreamingResponse(
                bypass_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "X-Cache-Lookup": "BYPASS",
                    "X-Cache-Tier": "NONE",
                    "X-Cache-Similarity": "0.0",
                    "X-Lookup-Latency-ms": "0.00",
                    "X-RateLimit-Limit": str(rate_limiter.default_limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        raw_response = await llm_client.forward_chat_completion(payload)
        response.headers["X-Cache-Lookup"] = "BYPASS"
        response.headers["X-Cache-Tier"] = "NONE"
        response.headers["X-Cache-Similarity"] = "0.0"
        response.headers["X-Lookup-Latency-ms"] = "0.00"
        return raw_response

    # 3. L1 EXACT-MATCH FAST PATH (Zero Embedding Generation Overhead)
    exact_key = ExactMatchCache.generate_exact_key(
        prompt=user_prompt,
        model=request.model,
        system_prompt_hash=sys_hash,
        params_hash=p_hash,
        context_hash=c_hash,
    )
    t_start = time.perf_counter()
    l1_cached = exact_cache.lookup(exact_key)
    l1_latency = (time.perf_counter() - t_start) * 1000.0

    if l1_cached:
        telemetry.record_hit(latency_ms=l1_latency)
        if request.stream:
            return StreamingResponse(
                StreamingUtils.replay_cached_stream(
                    cached_response=l1_cached, model=request.model
                ),
                media_type="text/event-stream",
                headers={
                    "X-Cache-Lookup": "HIT",
                    "X-Cache-Tier": "L1-EXACT",
                    "X-Cache-Similarity": "1.0000",
                    "X-Lookup-Latency-ms": f"{l1_latency:.2f}",
                    "X-RateLimit-Limit": str(rate_limiter.default_limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        response.headers["X-Cache-Lookup"] = "HIT"
        response.headers["X-Cache-Tier"] = "L1-EXACT"
        response.headers["X-Cache-Similarity"] = "1.0000"
        response.headers["X-Lookup-Latency-ms"] = f"{l1_latency:.2f}"
        return l1_cached

    # 4. L2 SEMANTIC VECTOR CACHE (HNSW Nearest Neighbors)
    cache_result, vector, _, _, _ = await cache.lookup(
        messages=serialized_messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        similarity_threshold=x_similarity_threshold,
    )

    if cache_result.is_hit and cache_result.response_data:
        telemetry.record_hit(latency_ms=cache_result.lookup_latency_ms)
        # Track similarity score for drift metrics
        from app.api.admin import drift_tracker
        drift_tracker.record_hit_similarity(cache_result.similarity_score)

        # Hydrate L1 exact cache for subsequent identical requests
        exact_cache.store(exact_key, cache_result.response_data)

        if request.stream:
            return StreamingResponse(
                StreamingUtils.replay_cached_stream(
                    cached_response=cache_result.response_data,
                    model=request.model,
                ),
                media_type="text/event-stream",
                headers={
                    "X-Cache-Lookup": "HIT",
                    "X-Cache-Tier": "L2-SEMANTIC",
                    "X-Cache-Similarity": f"{cache_result.similarity_score:.4f}",
                    "X-Lookup-Latency-ms": f"{cache_result.lookup_latency_ms:.2f}",
                    "X-RateLimit-Limit": str(rate_limiter.default_limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        response.headers["X-Cache-Lookup"] = "HIT"
        response.headers["X-Cache-Tier"] = "L2-SEMANTIC"
        response.headers["X-Cache-Similarity"] = f"{cache_result.similarity_score:.4f}"
        response.headers["X-Lookup-Latency-ms"] = f"{cache_result.lookup_latency_ms:.2f}"
        return cache_result.response_data

    # 5. CACHE MISS with Anti-Stampede Barrier
    payload = request.model_dump(exclude_none=True)
    stampede_key = f"{p_hash}:{sys_hash}"
    is_leader, lock_token = await cache_lock.acquire_lock(stampede_key)

    try:
        if not is_leader:
            # Recheck L1 & L2 after follower wait
            retry_l1 = exact_cache.lookup(exact_key)
            if retry_l1:
                response.headers["X-Cache-Lookup"] = "HIT"
                response.headers["X-Cache-Tier"] = "L1-EXACT"
                response.headers["X-Cache-Similarity"] = "1.0000"
                response.headers["X-Lookup-Latency-ms"] = "1.00"
                return retry_l1

            retry_hit, _, _, _, _ = await cache.lookup(
                messages=serialized_messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                similarity_threshold=x_similarity_threshold,
            )
            if retry_hit.is_hit and retry_hit.response_data:
                telemetry.record_hit(latency_ms=retry_hit.lookup_latency_ms)
                response.headers["X-Cache-Lookup"] = "HIT"
                response.headers["X-Cache-Tier"] = "L2-SEMANTIC"
                response.headers["X-Cache-Similarity"] = f"{retry_hit.similarity_score:.4f}"
                response.headers["X-Lookup-Latency-ms"] = f"{retry_hit.lookup_latency_ms:.2f}"
                return retry_hit.response_data

        if request.stream:
            telemetry.record_miss(latency_ms=cache_result.lookup_latency_ms)

            async def stream_and_cache_generator():
                assembled_data = None
                async for chunk_str, full_payload in llm_client.stream_chat_completion(payload):
                    if chunk_str:
                        yield chunk_str
                    if full_payload:
                        assembled_data = full_payload

                if assembled_data:
                    safe, _ = CacheSecurityGuardrail.is_cacheable(user_prompt, assembled_data)
                    if safe:
                        ttl = NegativeCachePolicy.get_adaptive_ttl(assembled_data, 86400)
                        cache.store(
                            prompt=user_prompt,
                            vector=vector,
                            response_data=assembled_data,
                            model=request.model,
                            system_prompt_hash=sys_hash,
                            params_hash=p_hash,
                            context_hash=c_hash,
                            ttl_seconds=ttl,
                        )
                        exact_cache.store(exact_key, assembled_data, ttl_seconds=ttl)

            return StreamingResponse(
                stream_and_cache_generator(),
                media_type="text/event-stream",
                headers={
                    "X-Cache-Lookup": "MISS",
                    "X-Cache-Tier": "NONE",
                    "X-Cache-Similarity": "0.0",
                    "X-Lookup-Latency-ms": f"{cache_result.lookup_latency_ms:.2f}",
                    "X-RateLimit-Limit": str(rate_limiter.default_limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        start_llm = time.perf_counter()
        raw_response = await llm_client.forward_chat_completion(payload)
        llm_latency_ms = (time.perf_counter() - start_llm) * 1000.0

        telemetry.record_miss(latency_ms=llm_latency_ms)

        safe, _ = CacheSecurityGuardrail.is_cacheable(user_prompt, raw_response)
        if safe:
            ttl = NegativeCachePolicy.get_adaptive_ttl(raw_response, 86400)
            cache.store(
                prompt=user_prompt,
                vector=vector,
                response_data=raw_response,
                model=request.model,
                system_prompt_hash=sys_hash,
                params_hash=p_hash,
                context_hash=c_hash,
                ttl_seconds=ttl,
            )
            exact_cache.store(exact_key, raw_response, ttl_seconds=ttl)

        response.headers["X-Cache-Lookup"] = "MISS"
        response.headers["X-Cache-Tier"] = "NONE"
        response.headers["X-Cache-Similarity"] = "0.0"
        response.headers["X-Lookup-Latency-ms"] = f"{cache_result.lookup_latency_ms:.2f}"
        return raw_response
    finally:
        if is_leader and lock_token:
            cache_lock.release_lock(stampede_key, lock_token)