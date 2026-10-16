from fastapi import APIRouter
from app.core.telemetry import TelemetryTracker

router = APIRouter(prefix="/v1/analytics", tags=["Analytics"])
telemetry = TelemetryTracker()


@router.get("/stats")
def get_cache_statistics():
    """Retrieve aggregate cache metrics, hit ratios, and estimated time savings."""
    return telemetry.get_stats()


@router.post("/reset")
def reset_cache_statistics():
    """Reset telemetry metrics counters."""
    telemetry.reset_stats()
    return {"status": "success", "message": "Telemetry metrics reset successfully"}