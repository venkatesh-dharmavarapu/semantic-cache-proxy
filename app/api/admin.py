from typing import Optional
from fastapi import APIRouter, Body, Query, Response, status
from app.api.proxy import cache
from app.api.proxy import session_manager
from app.core.drift_detector import SemanticDriftTracker
from app.core.feedback_manager import FeedbackManager
from app.core.tenant_policy import TenantPurgeService
from app.api.proxy import event_mesh
from app.schemas.event_models import InvalidationScope


@router.delete("/cache/tenants/{tenant_id}", status_code=status.HTTP_200_OK)
async def purge_tenant_cache(tenant_id: str):
    client = cache.vector_store.client
    if not client:
        return {"status": "error", "message": "Redis client not connected"}

    purged_count = TenantPurgeService.purge_tenant_namespace(client, tenant_id)
    # Broadcast tenant purge to mesh
    await event_mesh.publish_event(
        scope=InvalidationScope.TENANT_PURGED,
        target=tenant_id,
        metadata={"purged_count": purged_count},
    )
    return {
        "status": "success",
        "tenant_id": tenant_id,
        "purged_keys_count": purged_count,
        "message": f"Tenant '{tenant_id}' namespace successfully cleared.",
    }


@router.post("/cache/threshold/tune", status_code=status.HTTP_200_OK)
async def tune_dynamic_threshold(target_confidence: float = Body(default=0.88, embed=True)):
    stats = drift_tracker.get_distribution_stats()
    if stats["sample_count"] < 5:
        return {
            "status": "insufficient_data",
            "current_threshold": cache.similarity_threshold,
            "recommended_threshold": cache.similarity_threshold,
        }

    mean_val = stats["mean"]
    std_dev = stats["std_dev"]
    recommended = round(max(0.78, min(0.95, mean_val - (std_dev * 1.5))), 4)
    cache.similarity_threshold = recommended

    # Broadcast threshold change to all instances
    await event_mesh.publish_event(
        scope=InvalidationScope.THRESHOLD_UPDATED,
        target="global",
        metadata={"new_threshold": recommended},
    )

    return {
        "status": "updated",
        "sample_count": stats["sample_count"],
        "mean_similarity": mean_val,
        "new_threshold": recommended,
    }


@router.delete("/cache/purge", status_code=status.HTTP_200_OK)
async def purge_cache(
    model: Optional[str] = Query(default=None, description="Purge entries strictly for a model tag"),
    flush_all: bool = Query(default=False, description="Purge all entries in the cache index"),
):
    client = cache.vector_store.client
    if not client:
        return {"status": "error", "message": "Vector store client not connected"}

    purged_count = 0

    if flush_all:
        for key in client.scan_iter("cache:*"):
            client.delete(key)
            purged_count += 1
        await event_mesh.publish_event(scope=InvalidationScope.FLUSH_ALL, target="all")
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
        await event_mesh.publish_event(scope=InvalidationScope.MODEL_PURGED, target=model)
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

from app.schemas.feedback_models import (
    FeedbackStatsResponse,
    FeedbackSubmitRequest,
    FeedbackSubmitResponse,
)
from app.schemas.migration_models import (
    BatchIngestionRequest,
    BatchIngestionResponse,
    SchemaMigrationRequest,
    SchemaMigrationResponse,
)
from app.schemas.warming_models import CacheWarmRequest, CacheWarmResponse
from app.services.migration_engine import CacheMigrationEngine
from app.services.topology import ClusterTopologyInspector
from app.services.warmer import CacheWarmer

router = APIRouter(prefix="/v1", tags=["Administration & Governance"])
warmer = CacheWarmer(cache)
topology_inspector = ClusterTopologyInspector(cache.vector_store)
drift_tracker = SemanticDriftTracker(cache.vector_store)
feedback_manager = FeedbackManager(cache.vector_store)
migration_engine = CacheMigrationEngine(cache)


# ---------------------------
# Bulk Ingestion & Migration
# ---------------------------

@router.post("/cache/ingest/batch", response_model=BatchIngestionResponse, status_code=status.HTTP_200_OK)
async def ingest_batch_logs(request: BatchIngestionRequest):
    """Bulk-loads historical query logs or synthetic benchmarks directly into the vector index."""
    raw_dicts = [rec.model_dump() for rec in request.records]
    successes, failures = await migration_engine.ingest_batch_records(
        records=raw_dicts, chunk_size=request.chunk_size
    )
    return BatchIngestionResponse(
        status="completed",
        total_received=len(request.records),
        successfully_indexed=successes,
        failures=failures,
    )


@router.post("/cache/migrate", response_model=SchemaMigrationResponse, status_code=status.HTTP_200_OK)
def trigger_schema_migration(request: SchemaMigrationRequest):
    """Prepares schema migration and inspects cutover targets across versioned vector indices."""
    res = migration_engine.migrate_index_schema(new_index_name=request.target_index_name)
    return SchemaMigrationResponse(
        status=res.get("status", "error"),
        current_index=res.get("current_index", ""),
        target_index=res.get("target_index", ""),
        total_keys_ready_for_cutover=res.get("total_keys_ready_for_cutover", 0),
    )


# ---------------------------
# Feedback & Quality Endpoints
# ---------------------------

@router.post("/feedback", response_model=FeedbackSubmitResponse, status_code=status.HTTP_200_OK)
def submit_feedback(request: FeedbackSubmitRequest):
    """Submits user feedback (thumbs up / thumbs down) for a cached completion."""
    success, score, evicted = feedback_manager.record_feedback(
        cache_key=request.cache_key,
        completion_id=request.completion_id,
        rating=request.rating,
        reason=request.feedback_reason,
        user_comment=request.user_comment,
    )
    if not success:
        return FeedbackSubmitResponse(
            status="ignored",
            message="Unable to record feedback. Provide a valid cache_key or completion_id.",
            target_key=request.cache_key or request.completion_id,
            current_net_score=score,
            evicted=False,
        )

    msg = "Feedback registered successfully."
    if evicted:
        msg = "Feedback registered. Entry breached negative threshold and was evicted."

    return FeedbackSubmitResponse(
        status="success",
        message=msg,
        target_key=request.cache_key or request.completion_id,
        current_net_score=score,
        evicted=evicted,
    )


@router.get("/cache/feedback/stats", response_model=FeedbackStatsResponse, status_code=status.HTTP_200_OK)
def get_feedback_statistics():
    """Retrieves overall user satisfaction rates, ratings volume, and eviction tallies."""
    return feedback_manager.get_feedback_stats()


# ---------------------------
# Cache Management & Warmer
# ---------------------------

@router.post("/cache/warm", response_model=CacheWarmResponse, status_code=status.HTTP_200_OK)
async def warm_cache(request: CacheWarmRequest):
    """Pre-seeds the Redis semantic cache with batches of high-frequency prompts."""
    successes, failures = await warmer.warm_batch(request)
    return CacheWarmResponse(
        total_submitted=len(request.items),
        successfully_warmed=successes,
        failures=failures,
    )


@router.get("/cache/clusters", status_code=status.HTTP_200_OK)
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


@router.get("/cache/drift", status_code=status.HTTP_200_OK)
def get_semantic_drift():
    """Retrieves rolling similarity metrics, standard deviation, and drift alert status."""
    return drift_tracker.get_distribution_stats()


@router.get("/cache/topology", status_code=status.HTTP_200_OK)
def get_topology(max_clusters: int = Query(default=30, ge=1, le=100)):
    """Retrieves cluster centroid hierarchy, alias density, and dispersion distribution."""
    return topology_inspector.get_topology_map(max_clusters=max_clusters)


@router.post("/cache/threshold/tune", status_code=status.HTTP_200_OK)
def tune_dynamic_threshold(target_confidence: float = Body(default=0.88, embed=True)):
    """Dynamically recalculates runtime similarity threshold based on rolling drift."""
    stats = drift_tracker.get_distribution_stats()
    if stats["sample_count"] < 5:
        return {
            "status": "insufficient_data",
            "current_threshold": cache.similarity_threshold,
            "recommended_threshold": cache.similarity_threshold,
        }

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


@router.delete("/cache/purge", status_code=status.HTTP_200_OK)
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

@router.post("/chat/sessions/prune", status_code=status.HTTP_200_OK)
def prune_session_tree(
    session_id: str = Body(..., embed=True),
    after_turn: int = Body(..., embed=True),
):
    """Prunes and invalidates conversational cache turns that occurred after an edited turn index."""
    invalidated_count = session_manager.prune_branches_after_turn(
        session_id=session_id, turn_index=after_turn
    )
    return {
        "status": "success",
        "session_id": session_id,
        "pruned_after_turn": after_turn,
        "invalidated_turns_count": invalidated_count,
    }


@router.delete("/cache/tenants/{tenant_id}", status_code=status.HTTP_200_OK)
def purge_tenant_cache(tenant_id: str):
    """
    Purges all exact keys, vector entries, budgets, and rate limits
    belonging exclusively to the designated tenant.
    """
    client = cache.vector_store.client
    if not client:
        return {"status": "error", "message": "Redis client not connected"}

    purged_count = TenantPurgeService.purge_tenant_namespace(client, tenant_id)
    return {
        "status": "success",
        "tenant_id": tenant_id,
        "purged_keys_count": purged_count,
        "message": f"Tenant '{tenant_id}' namespace successfully cleared.",
    }