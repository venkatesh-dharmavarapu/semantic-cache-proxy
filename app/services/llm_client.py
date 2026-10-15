import logging
from typing import Any, Dict, Optional
import httpx
from app.config import settings
from app.core.exceptions import UpstreamProviderError, UpstreamTimeoutError

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ):
        base = base_url or settings.OLLAMA_BASE_URL
        self.base_url = f"{base.rstrip('/')}/v1"
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    async def forward_chat_completion(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Forwards completion request to local Ollama with retry backoff."""
        transport = httpx.AsyncHTTPTransport(retries=self.max_retries)
        async with httpx.AsyncClient(
            transport=transport, timeout=self.timeout
        ) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                logger.error("Upstream Ollama request timed out: %s", exc)
                raise UpstreamTimeoutError(
                    "Ollama model execution timed out."
                ) from exc
            except httpx.HTTPStatusError as exc:
                logger.error("Upstream returned HTTP error %s: %s", exc.response.status_code, exc.response.text)
                raise UpstreamProviderError(
                    f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                logger.error("Connection error to upstream Ollama: %s", exc)
                raise UpstreamProviderError(
                    "Unable to connect to local Ollama instance. Is Ollama running?"
                ) from exc