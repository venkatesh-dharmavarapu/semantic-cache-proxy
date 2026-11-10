from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.admin import router as admin_router
from app.api.analytics import router as analytics_router
from app.api.governance import router as governance_router
from app.api.health import router as health_router
from app.api.proxy import event_mesh, router as proxy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start Redis Pub/Sub listener daemon
    await event_mesh.start_listener()
    yield
    # Shutdown: Stop listener and release resources
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