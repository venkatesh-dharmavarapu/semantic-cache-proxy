from prometheus_client import Counter, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
from typing import Optional


# 1. Request Counter
CACHE_REQUESTS_TOTAL = Counter(
    "semantic_cache_requests_total",
    "Total chat completion requests handled by the proxy",
    ["tier", "status", "tenant_id", "model"],
)

# 2. Stage Lookup Latency Histogram
CACHE_LOOKUP_LATENCY_SECONDS = Histogram(
    "semantic_cache_lookup_latency_seconds",
    "Time spent in cache resolution pipeline (in seconds)",
    ["stage", "tier"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# 3. Token Savings Accumulators
CACHE_TOKENS_SAVED_TOTAL = Counter(
    "semantic_cache_tokens_saved_total",
    "Total tokens prevented from being dispatched to upstream models",
    ["token_type", "tenant_id", "model"],
)

CACHE_COST_SAVINGS_USD_TOTAL = Counter(
    "semantic_cache_cost_savings_usd_total",
    "Estimated financial savings achieved via cache hits in USD",
    ["tenant_id", "model"],
)

# 4. Semantic Similarity Distribution
CACHE_SIMILARITY_SCORE = Summary(
    "semantic_cache_similarity_score",
    "Observed cosine similarity distribution on L2 semantic cache hits",
    ["tenant_id", "model"],
)


class MetricsRecorder:
    @staticmethod
    def record_request(
        tier: str,
        status: str,
        tenant_id: str,
        model: str,
    ):
        CACHE_REQUESTS_TOTAL.labels(
            tier=tier,
            status=status,
            tenant_id=tenant_id,
            model=model,
        ).inc()

    @staticmethod
    def record_latency(stage: str, tier: str, latency_seconds: float):
        CACHE_LOOKUP_LATENCY_SECONDS.labels(
            stage=stage,
            tier=tier,
        ).observe(latency_seconds)

    @staticmethod
    def record_savings(
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
    ):
        if prompt_tokens > 0:
            CACHE_TOKENS_SAVED_TOTAL.labels(
                token_type="prompt",
                tenant_id=tenant_id,
                model=model,
            ).inc(prompt_tokens)

        if completion_tokens > 0:
            CACHE_TOKENS_SAVED_TOTAL.labels(
                token_type="completion",
                tenant_id=tenant_id,
                model=model,
            ).inc(completion_tokens)

        if estimated_cost_usd > 0.0:
            CACHE_COST_SAVINGS_USD_TOTAL.labels(
                tenant_id=tenant_id,
                model=model,
            ).inc(estimated_cost_usd)

    @staticmethod
    def record_similarity(tenant_id: str, model: str, score: float):
        CACHE_SIMILARITY_SCORE.labels(
            tenant_id=tenant_id,
            model=model,
        ).observe(score)

    @staticmethod
    def export_metrics() -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST