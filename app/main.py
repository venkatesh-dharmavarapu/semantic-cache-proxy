from fastapi import FastAPI
from app.api.admin import router as admin_router
from app.api.analytics import router as analytics_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.proxy import router as proxy_router
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Drop-in semantic caching reverse proxy for LLM APIs",
    version="1.0.0",
)

app.include_router(proxy_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(health_router)
app.include_router(metrics_router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}