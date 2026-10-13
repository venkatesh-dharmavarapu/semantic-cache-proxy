from typing import List
import httpx
from app.config import settings


class EmbeddingService:
    def __init__(
        self, base_url: str | None = None, model: str | None = None
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.EMBEDDING_MODEL

    async def get_embedding(self, text: str) -> List[float]:
        """Generates an embedding vector using local Ollama model."""
        cleaned_text = text.replace("\n", " ").strip()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": cleaned_text},
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation sequentially or concurrently."""
        embeddings = []
        for text in texts:
            vec = await self.get_embedding(text)
            embeddings.append(vec)
        return embeddings