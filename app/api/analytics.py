from fastapi import APIRouter
from app.core.sla_tracker import SLATracker
from app.core.telemetry import TelemetryTracker

router = APIRouter(prefix="/v1/analytics", tags=["Analytics"])
telemetry = TelemetryTracker()
sla_tracker = SLATracker()


@router.get("/stats")
def get_cache_statistics():
    """Retrieve aggregate cache metrics, hit ratios, and estimated time savings."""
    return telemetry.get_stats()


@router.get("/sla")
def get_sla_latency_breakdown():
    """Retrieve rolling P50, P90, P95, and P99 latency percentiles for hits and misses."""
    return sla_tracker.get_sla_metrics()


@router.post("/reset")
def reset_cache_statistics():
    """Reset telemetry metrics and SLA counters."""
    telemetry.reset_stats()
    sla_tracker.reset()
    return {"status": "success", "message": "Telemetry and SLA metrics reset successfully"}