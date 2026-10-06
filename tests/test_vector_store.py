import pytest
from app.core.vector_store import VectorStoreManager

def test_redis_connection_and_index():
    manager = VectorStoreManager()
    manager.connect()
    # Ping Redis
    assert manager.client.ping() is True
    
    # Create or ensure index exists
    manager.create_index(overwrite=False)
    assert manager.check_exists() is True