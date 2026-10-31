from typing import Optional
from fastapi import APIRouter, Body, Query, Response, status
from app.api.proxy import cache
from app.core.drift_detector import SemanticDriftTracker
from app.schemas.warming_models import CacheWarmRequest, CacheWarmResponse
from app.services.topology import ClusterTopologyInspector
from app.services.warmer import CacheWarmer

router = APIRouter(prefix="/v1/cache", tags=["Cache Administration"])
warmer = CacheWarmer(cache)
topology_inspector = ClusterTopologyInspector(cache.vector_store)
drift_tracker = SemanticDriftTracker(cache.vector_store)


@router.post("/warm", response_model=CacheWarmResponse, status_code=status.HTTP_200_OK)
async def warm_cache(request: CacheWarmRequest):
    """Pre-seeds the Redis semantic cache with batches of high-frequency prompts."""
    successes, failures = await warmer.warm_batch(request)
    return CacheWarmResponse(
        total_submitted=len(request.items),
        successfully_warmed=successes,
        failures=failures,
    )


@router.get("/clusters", status_code=status.HTTP_200_OK)
def get_cluster_stats():
    """Inspects cluster density, aliased prompts, and vector index memory distribution."""
    client = cache.vector_store.client
    if not client:
        return {"status": "error", "message": "Vector store client not connected"}

    total_keys = 0
    total_aliases = 0
    models_breakdown = {}

    for key in client.scan_iter("cache:llm:*"):
        total_keys += 1
        data = client.hmget(key, ["model", "alias_count", "hit_count"])
        model_name = data[0].decode("utf-8") if data[0] else "unknown"
        alias_cnt = int(data[1]) if data[1] else 0

        total_aliases += alias_cnt
        models_breakdown[model_name] = models_breakdown.get(model_name, 0) + 1

    return {
        "status": "success",
        "total_clusters": total_keys,
        "consolidated_aliases": total_aliases,
        "total_prompts_servable": total_keys + total_aliases,
        "models": models_breakdown,
    }


@router.get("/drift", status_code=status.HTTP_200_OK)
def get_semantic_drift():
    """Retrieves rolling similarity metrics, standard deviation, and drift alert status."""
    return drift_tracker.get_distribution_stats()


@router.get("/topology", status_code=status.HTTP_200_OK)
def get_topology(max_clusters: int = Query(default=30, ge=1, le=100)):
    """Retrieves cluster centroid hierarchy, alias density, and dispersion distribution."""
    return topology_inspector.get_topology_map(max_clusters=max_clusters)


@router.post("/threshold/tune", status_code=status.HTTP_200_OK)
def tune_dynamic_threshold(target_confidence: float = Body(default=0.88, embed=True)):
    """Dynamically recalculates runtime similarity threshold based on rolling drift."""
    stats = drift_tracker.get_distribution_stats()
    if stats["sample_count"] < 5:
        return {
            "status": "insufficient_data",
            "current_threshold": cache.similarity_threshold,
            "recommended_threshold": cache.similarity_threshold,
        }

    # If mean similarity is tightly clustered with low variance, safely lower threshold to capture more hits
    mean_val = stats["mean"]
    std_dev = stats["std_dev"]

    recommended = round(max(0.78, min(0.95, mean_val - (std_dev * 1.5))), 4)
    cache.similarity_threshold = recommended

    return {
        "status": "updated",
        "sample_count": stats["sample_count"],
        "mean_similarity": mean_val,
        "new_threshold": recommended,
    }


@router.delete("/purge", status_code=status.HTTP_200_OK)
def purge_cache(
    model: Optional[str] = Query(default=None, description="Purge entries strictly for a model tag"),
    flush_all: bool = Query(default=False, description="Purge all entries in the cache index"),
):
    """Purges cached entries by model tag or flushes entire semantic cache."""
    client = cache.vector_store.client
    if not client:
        return {"status": "error", "message": "Vector store client not connected"}

    purged_count = 0

    if flush_all:
        for key in client.scan_iter("cache:*"):
            client.delete(key)
            purged_count += 1
        return {
            "status": "success",
            "message": "Entire cache flushed successfully",
            "purged_count": purged_count,
        }

    if model:
        for key in client.scan_iter("cache:llm:*"):
            stored_model = client.hget(key, "model")
            if stored_model and stored_model.decode("utf-8") == model:
                client.delete(key)
                purged_count += 1
        return {
            "status": "success",
            "message": f"Cache entries for model '{model}' purged",
            "purged_count": purged_count,
        }

    return {
        "status": "ignored",
        "message": "Specify either 'model' or 'flush_all=true' to execute purge",
        "purged_count": 0,
    }