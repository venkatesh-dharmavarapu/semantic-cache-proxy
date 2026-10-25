from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import httpx
from app.config import settings
from app.core.vector_store import VectorStoreManager

router = APIRouter(prefix="/health", tags=["Health"])
store = VectorStoreManager()


@router.get("/live", status_code=status.HTTP_200_OK)
def liveness_probe():
    """Shallow probe indicating proxy process is active."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness_probe():
    """
    Deep readiness check inspecting:
    1. Redis Stack connection and index presence.
    2. Ollama embedding server responsiveness.
    """
    checks = {"redis": False, "ollama": False, "index_ready": False}
    status_code = status.HTTP_200_OK

    # 1. Verify Redis Connection & Index
    try:
        store.connect()
        if store.client and store.client.ping():
            checks["redis"] = True
            indices = store.client.execute_command("FT._LIST")
            decoded_indices = [idx.decode("utf-8") if isinstance(idx, bytes) else idx for idx in indices]
            if settings.CACHE_INDEX_NAME in decoded_indices:
                checks["index_ready"] = True
    except Exception:
        checks["redis"] = False

    # 2. Verify Ollama Service
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                checks["ollama"] = True
    except Exception:
        checks["ollama"] = False

    if not all(checks.values()):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if status_code == 200 else "unhealthy",
            "dependencies": checks,
        },
    )