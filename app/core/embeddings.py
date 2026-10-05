from typing import List
from openai import AsyncOpenAI
from app.config import settings

class EmbeddingService:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = model or settings.EMBEDDING_MODEL

    async def get_embedding(self, text: str) -> List[float]:
        """
        Embeds a single prompt string into a high-dimensional vector.
        Normalizes linebreaks for consistent embedding spaces.
        """
        cleaned_text = text.replace("\n", " ").strip()
        response = await self.client.embeddings.create(
            input=[cleaned_text],
            model=self.model,
        )
        return response.data[0].embedding

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation for bulk load or evaluation."""
        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        response = await self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]