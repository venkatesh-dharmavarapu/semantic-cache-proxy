import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.admin import router as admin_router
from app.api.analytics import router as analytics_router
from app.api.governance import router as governance_router
from app.api.health import lifecycle_state, router as health_router
from app.api.proxy import event_mesh, load_shedder, router as proxy_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing background daemons: EventMesh and LoadShedder...")
    await event_mesh.start_listener()
    await load_shedder.start_monitor()
    lifecycle_state.is_ready = True
    lifecycle_state.is_draining = False

    yield

    # Shutdown / Draining Phase
    logger.info("SIGTERM received: Switching readiness to DRAINING...")
    lifecycle_state.is_ready = False
    lifecycle_state.is_draining = True

    # Allow active in-flight requests to complete (drain grace period)
    drain_timeout = 5.0
    start_drain = asyncio.get_event_loop().time()
    while load_shedder.current_in_flight > 0:
        if asyncio.get_event_loop().time() - start_drain > drain_timeout:
            logger.warning("Drain timeout exceeded; forcing shutdown.")
            break
        await asyncio.sleep(0.1)

    logger.info("Releasing event mesh and monitor connections...")
    await load_shedder.stop_monitor()
    await event_mesh.close()


app = FastAPI(
    title="Production Semantic Cache Proxy",
    version="1.0.0",
    description="Tiered Exact & Semantic HNSW Vector Cache Proxy for LLMs",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(proxy_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(governance_router)