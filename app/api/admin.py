from typing import Optional
from fastapi import APIRouter, Query, Response, status
from app.api.proxy import cache
from app.schemas.warming_models import CacheWarmRequest, CacheWarmResponse
from app.services.warmer import CacheWarmer

router = APIRouter(prefix="/v1/cache", tags=["Cache Administration"])
warmer = CacheWarmer(cache)


@router.post("/warm", response_model=CacheWarmResponse, status_code=status.HTTP_200_OK)
async def warm_cache(request: CacheWarmRequest):
    """Pre-seeds the Redis semantic cache with batches of high-frequency prompts."""
    successes, failures = await warmer.warm_batch(request)
    return CacheWarmResponse(
        total_submitted=len(request.items),
        successfully_warmed=successes,
        failures=failures,
    )


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
        for key in client.scan_iter("cache:llm:*"):
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