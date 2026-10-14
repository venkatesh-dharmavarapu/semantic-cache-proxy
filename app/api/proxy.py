from fastapi import APIRouter, Header, Response
from app.core.semantic_cache import SemanticCache
from app.schemas.chat_models import ChatCompletionRequest, ChatCompletionResponse
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/v1", tags=["Chat Proxy"])

cache = SemanticCache()
llm_client = LLMClient()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    x_similarity_threshold: float | None = Header(default=None),
):
    serialized_messages = [msg.model_dump() for msg in request.messages]

    # 1. Semantic cache check
    cache_result, vector, sys_hash, p_hash = await cache.lookup(
        messages=serialized_messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        similarity_threshold=x_similarity_threshold,
    )

    # 2. Return cached response on HIT
    if cache_result.is_hit and cache_result.response_data:
        response.headers["X-Cache-Lookup"] = "HIT"
        response.headers["X-Cache-Similarity"] = f"{cache_result.similarity_score:.4f}"
        response.headers["X-Lookup-Latency-ms"] = f"{cache_result.lookup_latency_ms:.2f}"
        return cache_result.response_data

    # 3. Cache MISS: Forward to upstream Ollama
    payload = request.model_dump(exclude_none=True)
    raw_response = await llm_client.forward_chat_completion(payload)

    # 4. Store miss in Redis vector store
    user_prompt = serialized_messages[-1]["content"] if serialized_messages else ""
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