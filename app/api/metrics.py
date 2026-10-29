from fastapi import APIRouter, Response
from app.core.telemetry import TelemetryTracker

router = APIRouter(tags=["Metrics"])
telemetry = TelemetryTracker()


@router.get("/metrics")
def prometheus_metrics():
    """Generates Prometheus text format scrape metrics."""
    stats = telemetry.get_stats()

    total = stats.get("total_requests", 0)
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    bypasses = stats.get("bypasses", 0)
    saved_ms = stats.get("saved_latency_ms", 0.0)

    lines = [
        "# HELP semantic_cache_requests_total Total number of incoming requests to semantic cache",
        "# TYPE semantic_cache_requests_total counter",
        f'semantic_cache_requests_total{{status="hit"}} {hits}',
        f'semantic_cache_requests_total{{status="miss"}} {misses}',
        f'semantic_cache_requests_total{{status="bypass"}} {bypasses}',
        f'semantic_cache_requests_total{{status="all"}} {total}',
        "",
        "# HELP semantic_cache_saved_latency_seconds_total Total estimated latency saved by cache hits",
        "# TYPE semantic_cache_saved_latency_seconds_total counter",
        f"semantic_cache_saved_latency_seconds_total {round(saved_ms / 1000.0, 4)}",
        "",
        "# HELP semantic_cache_hit_ratio_percent Percentage of cache hits over total requests",
        "# TYPE semantic_cache_hit_ratio_percent gauge",
        f"semantic_cache_hit_ratio_percent {stats.get('hit_ratio_percent', 0.0)}",
        "",
    ]

    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4")