from typing import Any, Dict
import httpx
from app.config import settings


class LLMClient:
    def __init__(self, base_url: str | None = None):
        base = base_url or settings.OLLAMA_BASE_URL
        self.base_url = f"{base.rstrip('/')}/v1"

    async def forward_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Forwards completion request to local Ollama /v1 endpoint."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()