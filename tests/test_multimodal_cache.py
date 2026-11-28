import base64
import io
from PIL import Image
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.api.proxy import cache, llm_client
from app.core.multimodal_hasher import PerceptualImageHasher


@pytest.fixture(autouse=True)
def clean_redis():
    client = cache.vector_store.client
    for key in client.scan_iter("cache:*"):
        client.delete(key)
    try:
        cache.vector_store.index.delete(drop=True)
    except Exception:
        pass
    cache.vector_store.create_index(overwrite=True)
    yield
    for key in client.scan_iter("cache:*"):
        client.delete(key)


def _generate_test_image_b64(color: str, size: tuple = (64, 64)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def test_dhash_invariance_to_resize():
    # Generate 128x128 vs 64x64 image of the same gradient pattern
    img1 = Image.linear_gradient("L").resize((128, 128))
    img2 = Image.linear_gradient("L").resize((64, 64))

    buf1, buf2 = io.BytesIO(), io.BytesIO()
    img1.save(buf1, format="PNG")
    img2.save(buf2, format="JPEG", quality=85)

    hash1 = PerceptualImageHasher.compute_dhash(buf1.getvalue())
    hash2 = PerceptualImageHasher.compute_dhash(buf2.getvalue())

    # Hashes must be identical or have minimal Hamming distance
    dist = PerceptualImageHasher.hamming_distance(hash1, hash2)
    assert dist <= 2


@pytest.mark.asyncio
async def test_multimodal_cache_isolation_and_hit(monkeypatch):
    async def mock_embed(text: str):
        return [0.05] * 768

    monkeypatch.setattr(cache.embedder, "get_embedding", mock_embed)

    call_count = 0

    async def mock_forward(payload, traceparent=None):
        nonlocal call_count
        call_count += 1
        return {
            "id": f"chatcmpl-vlm-{call_count}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2-vision",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Image analysis {call_count}"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
        }

    monkeypatch.setattr(llm_client, "forward_chat_completion", mock_forward)

    b64_red_img = _generate_test_image_b64("red", (64, 64))
    b64_blue_img = _generate_test_image_b64("blue", (64, 64))

    req_red = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this visual element"},
                    {"type": "image_url", "image_url": {"url": b64_red_img}},
                ],
            }
        ],
        "temperature": 0.0,
    }

    req_blue = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this visual element"},
                    {"type": "image_url", "image_url": {"url": b64_blue_img}},
                ],
            }
        ],
        "temperature": 0.0,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Request 1: Red Image -> MISS
        res1 = await ac.post("/v1/chat/completions", json=req_red)
        assert res1.status_code == 200
        assert res1.headers["X-Cache-Lookup"] == "MISS"
        assert res1.headers.get("X-Is-MultiModal") == "true"
        assert call_count == 1
        red_hash = res1.headers.get("X-Image-Perceptual-Hash")

        # Request 2: Exact same Red Image -> HIT
        res2 = await ac.post("/v1/chat/completions", json=req_red)
        assert res2.status_code == 200
        assert res2.headers["X-Cache-Lookup"] == "HIT"
        assert call_count == 1

        # Request 3: Blue Image with identical text -> MUST MISS (Visual isolation)
        res3 = await ac.post("/v1/chat/completions", json=req_blue)
        assert res3.status_code == 200
        assert res3.headers["X-Cache-Lookup"] == "MISS"
        assert call_count == 2
        blue_hash = res3.headers.get("X-Image-Perceptual-Hash")
        assert red_hash != blue_hash