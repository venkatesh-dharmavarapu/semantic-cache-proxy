import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


class StreamingTeePipe:
    """
    Tees an upstream SSE chunk stream to the client while assembling the full
    completion payload in-memory for background caching.
    """

    @staticmethod
    def format_sse_chunk(
        content_delta: str,
        model: str,
        chunk_id: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        """Formats a text token into standard OpenAI-compatible SSE chunk payload."""
        data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content_delta} if content_delta else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(data)}\n\n"

    @classmethod
    async def stream_cached_response(
        cls,
        cached_response: Dict[str, Any],
        chunk_size_chars: int = 24,
        delay_seconds: float = 0.005,
    ) -> AsyncGenerator[str, None]:
        """
        Re-inflates a static cached response into an SSE token stream for speculative TTFT acceleration.
        """
        choices = cached_response.get("choices", [])
        if not choices:
            yield "data: [DONE]\n\n"
            return

        message = choices[0].get("message", {})
        full_content = message.get("content", "")
        model_name = cached_response.get("model", "cached-model")
        completion_id = cached_response.get("id", f"chatcmpl-stream-{int(time.time())}")

        # Stream initial role chunk
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        # Stream content tokens in chunks
        for i in range(0, len(full_content), chunk_size_chars):
            chunk_slice = full_content[i : i + chunk_size_chars]
            yield cls.format_sse_chunk(chunk_slice, model_name, completion_id)
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        # Stream termination chunk
        yield cls.format_sse_chunk("", model_name, completion_id, finish_reason="stop")
        yield "data: [DONE]\n\n"

    @classmethod
    async def tee_upstream_stream(
        cls,
        upstream_response: httpx.Response,
        on_complete_callback,
    ) -> AsyncGenerator[str, None]:
        """
        Forwards upstream SSE lines directly to the client while accumulating tokens.
        Calls on_complete_callback(assembled_response) upon completion.
        """
        accumulated_text: List[str] = []
        model_name = "llama3.2:1b"
        completion_id = f"chatcmpl-stream-{int(time.time())}"

        try:
            async for line in upstream_response.aiter_lines():
                if not line:
                    continue

                # Forward raw SSE line to client
                yield f"{line}\n\n"

                if line.strip() == "data: [DONE]":
                    break

                if line.startswith("data: "):
                    raw_data = line[len("data: ") :].strip()
                    try:
                        chunk_obj = json.loads(raw_data)
                        model_name = chunk_obj.get("model", model_name)
                        completion_id = chunk_obj.get("id", completion_id)
                        choices = chunk_obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated_text.append(content)
                    except json.JSONDecodeError:
                        continue

            assembled_content = "".join(accumulated_text)
            assembled_response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assembled_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(assembled_content.split()),
                    "total_tokens": len(assembled_content.split()),
                },
            }

            if asyncio.iscoroutinefunction(on_complete_callback):
                await on_complete_callback(assembled_response)
            else:
                on_complete_callback(assembled_response)

        except Exception as exc:
            logger.error("Error in streaming tee pipe: %s", exc)
            raise exc