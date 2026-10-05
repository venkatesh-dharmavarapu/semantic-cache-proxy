import pytest
from unittest.mock import AsyncMock, MagicMock
from app.core.embeddings import EmbeddingService

@pytest.mark.asyncio
async def test_get_embedding_formatting(monkeypatch):
    service = EmbeddingService(api_key="sk-dummy")
    
    mock_data = MagicMock()
    mock_data.embedding = [0.1] * 1536
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]
    
    mock_create = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(service.client.embeddings, "create", mock_create)
    
    vector = await service.get_embedding("What is an API?\n")
    
    assert len(vector) == 1536
    assert vector[0] == 0.1
    # Check that newlines were stripped
    mock_create.assert_called_once_with(
        input=["What is an API?"],
        model="text-embedding-3-small"
    )