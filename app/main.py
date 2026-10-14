from fastapi import FastAPI
from app.api.proxy import router as proxy_router
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Drop-in semantic caching reverse proxy for LLM APIs",
    version="1.0.0",
)

app.include_router(proxy_router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}