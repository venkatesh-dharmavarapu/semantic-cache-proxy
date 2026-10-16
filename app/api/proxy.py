import time
from fastapi import APIRouter, Header, Response
from app.core.semantic_cache import SemanticCache
from app.core.telemetry import TelemetryTracker
from app.schemas.chat_models import ChatCompletionRequest, ChatCompletionResponse
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/v1", tags=["Chat Proxy"])

cache = SemanticCache()
llm_client = LLMClient()
telemetry = TelemetryTracker()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    x_cache_bypass: bool = Header(default=False),
    x_similarity_threshold: float | None = Header(default=None),
):
    serialized_messages = [msg.model_dump() for msg in request.messages]
    user_prompt = serialized_messages[-1]["content"] if serialized_messages else ""

    # 1. Handle Explicit Cache Bypass
    if x_cache_bypass:
        telemetry.record_bypass()
        payload = request.model_dump(exclude_none=True)
        raw_response = await llm_client.forward_chat_completion(payload)

        response.headers["X-Cache-Lookup"] = "BYPASS"
        response.headers["X-Cache-Similarity"] = "0.0"
        response.headers["X-Lookup-Latency-ms"] = "0.00"
        return raw_response

    # 2. Check Semantic Cache
    cache_result, vector, sys_hash, p_hash = await cache.lookup(
        messages=serialized_messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        similarity_threshold=x_similarity_threshold,
    )

    # 3. Cache HIT: Return immediately and record metrics
    if cache_result.is_hit and cache_result.response_data:
        telemetry.record_hit(latency_ms=cache_result.lookup_latency_ms)

        response.headers["X-Cache-Lookup"] = "HIT"
        response.headers["X-Cache-Similarity"] = f"{cache_result.similarity_score:.4f}"
        response.headers["X-Lookup-Latency-ms"] = f"{cache_result.lookup_latency_ms:.2f}"
        return cache_result.response_data

    # 4. Cache MISS: Forward to upstream Ollama
    start_llm = time.perf_counter()
    payload = request.model_dump(exclude_none=True)
    raw_response = await llm_client.forward_chat_completion(payload)
    llm_latency_ms = (time.perf_counter() - start_llm) * 1000.0

    telemetry.record_miss(latency_ms=llm_latency_ms)

    # Store entry in Redis
    cache.store(
        prompt=user_prompt,
        vector=vector,
        response_data=raw_response,
        model=request.model,
        system_prompt_hash=sys_hash,
        params_hash=p_hash,
    )

    response.headers["X-Cache-Lookup"] = "MISS"
    response.headers["X-Cache-Similarity"] = "0.0"
    response.headers["X-Lookup-Latency-ms"] = f"{cache_result.lookup_latency_ms:.2f}"

    return raw_response