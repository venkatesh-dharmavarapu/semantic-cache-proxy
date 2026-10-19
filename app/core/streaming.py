import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List


class StreamingUtils:
    @staticmethod
    def create_chunk_payload(
        request_id: str,
        model: str,
        delta_content: str,
        finish_reason: str | None = None,
    ) -> Dict[str, Any]:
        """Constructs an OpenAI-compliant chat.completion.chunk object."""
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta_content} if delta_content else {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    @staticmethod
    async def replay_cached_stream(
        cached_response: Dict[str, Any],
        model: str,
        chunk_size: int = 4,
        pacing_delay: float = 0.015,
    ) -> AsyncGenerator[str, None]:
        """
        Replays a cached completion payload as an SSE token stream.
        Paces output to mimic incremental generation while remaining fast.
        """
        request_id = cached_response.get("id") or f"chatcmpl-cached-{uuid.uuid4().hex[:8]}"
        content = (
            cached_response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        # Emit initial role delta
        initial_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        # Emit token chunks
        words = content.split(" ")
        for i in range(0, len(words), chunk_size):
            chunk_slice = words[i : i + chunk_size]
            text = " ".join(chunk_slice)
            if i + chunk_size < len(words):
                text += " "

            chunk_payload = StreamingUtils.create_chunk_payload(
                request_id=request_id,
                model=model,
                delta_content=text,
                finish_reason=None,
            )
            yield f"data: {json.dumps(chunk_payload)}\n\n"
            if pacing_delay > 0:
                await asyncio.sleep(pacing_delay)

        # Emit final closing chunk and terminator
        final_chunk = StreamingUtils.create_chunk_payload(
            request_id=request_id,
            model=model,
            delta_content="",
            finish_reason="stop",
        )
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"