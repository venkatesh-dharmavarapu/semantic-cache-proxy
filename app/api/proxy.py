import time
from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import StreamingResponse
from app.core.context_normalizer import ContextNormalizer
from app.core.distributed_lock import DistributedCacheLock
from app.core.exact_cache import ExactMatchCache
from app.core.exceptions import RateLimitExceededError
from app.core.hasher import CacheKeyHasher
from app.core.negative_cache import NegativeCachePolicy
from app.core.profiler import RequestProfiler
from app.core.rate_limiter import SlidingWindowRateLimiter
from app.core.security_guardrail import CacheSecurityGuardrail
from app.core.semantic_cache import SemanticCache
from app.core.session_manager import SessionThreadManager
from app.core.sla_tracker import SLATracker
from app.core.streaming import StreamingUtils
from app.core.telemetry import TelemetryTracker
from app.schemas.chat_models import ChatCompletionRequest
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/v1", tags=["Chat Proxy"])

cache = SemanticCache()
exact_cache = ExactMatchCache(cache.vector_store)
llm_client = LLMClient()
telemetry = TelemetryTracker()
sla_tracker = SLATracker(cache.vector_store)
rate_limiter = SlidingWindowRateLimiter(default_limit=60, window_seconds=60)
cache_lock = DistributedCacheLock(cache.vector_store)
session_manager = SessionThreadManager(cache.vector_store)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    response: Response,
    x_cache_bypass: bool = Header(default=False),
    x_similarity_threshold: float | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
    x_turn_index: int | None = Header(default=None),
):
    profiler = RequestProfiler()
    client_ip = (
        req.headers.get("x-forwarded-for")
        or (req.client.host if req.client else "127.0.0.1")
    )

    # 1. Rate limiter check
    profiler.start("rate_limit")
    allowed, remaining, retry_after = rate_limiter.check_rate_limit(client_ip)
    profiler.stop("rate_limit")
    if not allowed:
        raise RateLimitExceededError(
            retry_after=retry_after, limit=rate_limiter.default_limit
        )

    response.headers["X-RateLimit-Limit"] = str(rate_limiter.default_limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    profiler.start("hash")
    serialized_messages = [msg.model_dump() for msg in request.messages]
    user_prompt, system_prompt = cache._extract_prompt_and_system(serialized_messages)

    # Sliding-window multi-turn partition
    local_window, history_digest, _ = ContextNormalizer.partition_context(serialized_messages)
    c_hash = ContextNormalizer.compute_turn_context_hash(local_window, history_digest)

    sys_hash = CacheKeyHasher.hash_system_prompt(system_prompt)
    p_hash = CacheKeyHasher.hash_generation_params(
        model=request.model, temperature=request.temperature, max_tokens=request.max_tokens
    )
    profiler.stop("hash")

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
                    "Server-Timing": profiler.to_server_timing_header(),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        profiler.start("upstream")
        raw_response = await llm_client.forward_chat_completion(payload)
        profiler.stop("upstream")

        response.headers["X-Cache-Lookup"] = "BYPASS"
        response.headers["X-Cache-Tier"] = "NONE"
        response.headers["X-Cache-Similarity"] = "0.0"
        response.headers["X-Lookup-Latency-ms"] = "0.00"
        response.headers["Server-Timing"] = profiler.to_server_timing_header()
        return raw_response

    # 3. L1 EXACT-MATCH FAST PATH
    exact_key = ExactMatchCache.generate_exact_key(
        prompt=user_prompt,
        model=request.model,
        system_prompt_hash=sys_hash,
        params_hash=p_hash,
        context_hash=c_hash,
    )
    profiler.start("l1_exact")
    l1_cached = exact_cache.lookup(exact_key)
    profiler.stop("l1_exact")

    if l1_cached:
        total_lat = profiler.total_elapsed_ms()
        telemetry.record_hit(latency_ms=total_lat)
        sla_tracker.record_latency(is_hit=True, latency_ms=total_lat)

        if x_session_id and x_turn_index is not None:
            session_manager.record_turn(x_session_id, x_turn_index, exact_key)

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
                    "X-Lookup-Latency-ms": f"{total_lat:.2f}",
                    "Server-Timing": profiler.to_server_timing_header(),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        response.headers["X-Cache-Lookup"] = "HIT"
        response.headers["X-Cache-Tier"] = "L1-EXACT"
        response.headers["X-Cache-Similarity"] = "1.0000"
        response.headers["X-Lookup-Latency-ms"] = f"{total_lat:.2f}"
        response.headers["Server-Timing"] = profiler.to_server_timing_header()
        return l1_cached

    # 4. L2 SEMANTIC VECTOR CACHE
    profiler.start("l2_semantic")
    # Pass c_hash from multi-turn context
    vector = await cache.embedder.get_embedding(user_prompt)
    match = cache.reader.query_similarity(
        vector=vector,
        model=request.model,
        system_prompt_hash=sys_hash,
        params_hash=p_hash,
        context_hash=c_hash,
        similarity_threshold=x_similarity_threshold,
        raw_prompt=user_prompt,
    )
    profiler.stop("l2_semantic")

    if match:
        resp_data, similarity_score, entry_key = match
        total_lat = profiler.total_elapsed_ms()
        telemetry.record_hit(latency_ms=total_lat)
        sla_tracker.record_latency(is_hit=True, latency_ms=total_lat)

        from app.api.admin import drift_tracker
        drift_tracker.record_hit_similarity(similarity_score)

        cache.reader.extend_key_ttl(entry_key)
        exact_cache.store_exact(exact_key, resp_data)

        if x_session_id and x_turn_index is not None:
            session_manager.record_turn(x_session_id, x_turn_index, entry_key)

        if request.stream:
            return StreamingResponse(
                StreamingUtils.replay_cached_stream(
                    cached_response=resp_data,
                    model=request.model,
                ),
                media_type="text/event-stream",
                headers={
                    "X-Cache-Lookup": "HIT",
                    "X-Cache-Tier": "L2-SEMANTIC",
                    "X-Cache-Similarity": f"{similarity_score:.4f}",
                    "X-Lookup-Latency-ms": f"{total_lat:.2f}",
                    "Server-Timing": profiler.to_server_timing_header(),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        response.headers["X-Cache-Lookup"] = "HIT"
        response.headers["X-Cache-Tier"] = "L2-SEMANTIC"
        response.headers["X-Cache-Similarity"] = f"{similarity_score:.4f}"
        response.headers["X-Lookup-Latency-ms"] = f"{total_lat:.2f}"
        response.headers["Server-Timing"] = profiler.to_server_timing_header()
        return resp_data

    # 5. CACHE MISS with Anti-Stampede Barrier
    payload = request.model_dump(exclude_none=True)
    stampede_key = f"{p_hash}:{sys_hash}:{c_hash}"
    is_leader, lock_token = await cache_lock.acquire_lock(stampede_key)

    try:
        if not is_leader:
            retry_l1 = exact_cache.lookup(exact_key)
            if retry_l1:
                total_lat = profiler.total_elapsed_ms()
                response.headers["X-Cache-Lookup"] = "HIT"
                response.headers["X-Cache-Tier"] = "L1-EXACT"
                response.headers["X-Cache-Similarity"] = "1.0000"
                response.headers["X-Lookup-Latency-ms"] = f"{total_lat:.2f}"
                response.headers["Server-Timing"] = profiler.to_server_timing_header()
                return retry_l1

            retry_match = cache.reader.query_similarity(
                vector=vector,
                model=request.model,
                system_prompt_hash=sys_hash,
                params_hash=p_hash,
                context_hash=c_hash,
                similarity_threshold=x_similarity_threshold,
                raw_prompt=user_prompt,
            )
            if retry_match:
                r_data, r_sim, r_key = retry_match
                total_lat = profiler.total_elapsed_ms()
                telemetry.record_hit(latency_ms=total_lat)
                sla_tracker.record_latency(is_hit=True, latency_ms=total_lat)
                response.headers["X-Cache-Lookup"] = "HIT"
                response.headers["X-Cache-Tier"] = "L2-SEMANTIC"
                response.headers["X-Cache-Similarity"] = f"{r_sim:.4f}"
                response.headers["X-Lookup-Latency-ms"] = f"{total_lat:.2f}"
                response.headers["Server-Timing"] = profiler.to_server_timing_header()
                return r_data

        if request.stream:
            total_lat = profiler.total_elapsed_ms()
            telemetry.record_miss(latency_ms=total_lat)
            sla_tracker.record_latency(is_hit=False, latency_ms=total_lat)

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
                        stored_key = cache.store(
                            prompt=user_prompt,
                            vector=vector,
                            response_data=assembled_data,
                            model=request.model,
                            system_prompt_hash=sys_hash,
                            params_hash=p_hash,
                            context_hash=c_hash,
                            ttl_seconds=ttl,
                        )
                        exact_cache.store_exact(exact_key, assembled_data, ttl_seconds=ttl)
                        if x_session_id and x_turn_index is not None:
                            session_manager.record_turn(x_session_id, x_turn_index, stored_key)

            return StreamingResponse(
                stream_and_cache_generator(),
                media_type="text/event-stream",
                headers={
                    "X-Cache-Lookup": "MISS",
                    "X-Cache-Tier": "NONE",
                    "X-Cache-Similarity": "0.0",
                    "X-Lookup-Latency-ms": f"{total_lat:.2f}",
                    "Server-Timing": profiler.to_server_timing_header(),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        profiler.start("upstream")
        raw_response = await llm_client.forward_chat_completion(payload)
        profiler.stop("upstream")

        total_lat = profiler.total_elapsed_ms()
        telemetry.record_miss(latency_ms=total_lat)
        sla_tracker.record_latency(is_hit=False, latency_ms=total_lat)

        safe, _ = CacheSecurityGuardrail.is_cacheable(user_prompt, raw_response)
        if safe:
            ttl = NegativeCachePolicy.get_adaptive_ttl(raw_response, 86400)
            stored_key = cache.store(
                prompt=user_prompt,
                vector=vector,
                response_data=raw_response,
                model=request.model,
                system_prompt_hash=sys_hash,
                params_hash=p_hash,
                context_hash=c_hash,
                ttl_seconds=ttl,
            )
            exact_cache.store_exact(exact_key, raw_response, ttl_seconds=ttl)
            if x_session_id and x_turn_index is not None:
                session_manager.record_turn(x_session_id, x_turn_index, stored_key)

        response.headers["X-Cache-Lookup"] = "MISS"
        response.headers["X-Cache-Tier"] = "NONE"
        response.headers["X-Cache-Similarity"] = "0.0"
        response.headers["X-Lookup-Latency-ms"] = f"{total_lat:.2f}"
        response.headers["Server-Timing"] = profiler.to_server_timing_header()
        return raw_response
    finally:
        if is_leader and lock_token:
            cache_lock.release_lock(stampede_key, lock_token)