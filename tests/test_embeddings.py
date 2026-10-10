import pytest
from unittest.mock import AsyncMock, MagicMock
from app.core.embeddings import EmbeddingService
import httpx


@pytest.mark.asyncio
async def test_get_embedding_formatting(monkeypatch):
    service = EmbeddingService(base_url="http://localhost:11434", model="nomic-embed-text")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.1] * 768}
    mock_resp.raise_for_status.return_value = None

    mock_post = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    vector = await service.get_embedding("What is an API?\n")

    assert len(vector) == 768
    assert vector[0] == 0.1