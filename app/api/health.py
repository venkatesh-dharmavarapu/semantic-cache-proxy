from fastapi import APIRouter, Response, status
from app.core.metrics import MetricsRecorder

router = APIRouter(tags=["Health & Telemetry"])

# Global application lifecycle state
class AppLifecycleState:
    is_ready: bool = True
    is_draining: bool = False

lifecycle_state = AppLifecycleState()


@router.get("/health", status_code=status.HTTP_200_OK)
def legacy_health():
    return {"status": "healthy"}


@router.get("/health/live", status_code=status.HTTP_200_OK)
def liveness_probe():
    """Liveness probe: returns 200 as long as the process is alive."""
    return {"status": "alive"}


@router.get("/health/ready")
def readiness_probe(response: Response):
    """
    Readiness probe: asserts Redis connectivity and ensures
    the instance is not currently draining connections.
    """
    if lifecycle_state.is_draining or not lifecycle_state.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "draining", "ready": False}

    try:
        from app.api.proxy import cache
        client = cache.vector_store.client
        if client and client.ping():
            return {"status": "ready", "ready": True}
    except Exception as exc:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "ready": False, "error": str(exc)}

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "not_connected", "ready": False}


@router.get("/metrics", status_code=status.HTTP_200_OK)
def prometheus_metrics():
    """Scrapes system and cache telemetrics in OpenMetrics exposition format."""
    data, content_type = MetricsRecorder.export_metrics()
    return Response(content=data, media_type=content_type)