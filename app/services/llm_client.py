import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
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
        """Forwards standard completion request to local Ollama with retry backoff."""
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
                raise UpstreamTimeoutError("Ollama model execution timed out.") from exc
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

    async def stream_chat_completion(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[Tuple[str, Optional[Dict[str, Any]]], None]:
        """
        Streams SSE chunks from upstream Ollama to the client in real-time.
        Emits `(raw_chunk_line, assembled_completion_dict)` tuples.
        The completed response dictionary is only emitted on the terminal chunk.
        """
        payload_with_stream = dict(payload)
        payload_with_stream["stream"] = True

        accumulated_text: List[str] = []
        response_id: Optional[str] = None
        model_name: str = payload.get("model", "llama3.2:1b")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload_with_stream,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    error_text = await response.aread()
                    raise UpstreamProviderError(
                        f"Ollama stream error {response.status_code}: {error_text.decode('utf-8')}"
                    )

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # Forward current SSE line
                    yield f"{line}\n\n", None

                    # Intercept and buffer data content
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        raw_data = line[6:].strip()
                        try:
                            parsed = json.loads(raw_data)
                            if not response_id:
                                response_id = parsed.get("id")
                            choices = parsed.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    accumulated_text.append(token)
                        except json.JSONDecodeError:
                            continue

        # Assemble full canonical completion payload for the cache writer
        full_text = "".join(accumulated_text)
        assembled_payload = {
            "id": response_id or f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(payload.get("messages", [])),
                "completion_tokens": len(accumulated_text),
                "total_tokens": len(payload.get("messages", [])) + len(accumulated_text),
            },
        }

        # Final yield emitting the complete payload for caching
        yield "", assembled_payload