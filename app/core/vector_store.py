import logging
from typing import Optional
import redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.index: Optional[SearchIndex] = None

    def connect(self):
        if self.client is None:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD or None,
                decode_responses=False,
            )
            self._init_index()

    def _init_index(self):
        schema_dict = {
            "index": {
                "name": settings.CACHE_INDEX_NAME,
                "prefix": settings.CACHE_KEY_PREFIX,
                "storage_type": "hash",
            },
            "fields": [
                {"name": "prompt", "type": "text"},
                {"name": "response_json", "type": "text"},
                {"name": "model", "type": "tag"},
                {"name": "system_prompt_hash", "type": "tag"},
                {"name": "params_hash", "type": "tag"},
                {"name": "context_hash", "type": "tag"},
                {"name": "tenant_id", "type": "tag"},
                {"name": "created_at", "type": "numeric"},
                {"name": "ttl", "type": "numeric"},
                {"name": "hit_count", "type": "numeric"},
                {"name": "alias_count", "type": "numeric"},
                {
                    "name": "prompt_vector",
                    "type": "vector",
                    "attrs": {
                        "dims": settings.EMBEDDING_DIMENSION,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        }
        schema = IndexSchema.from_dict(schema_dict)
        self.index = SearchIndex(schema=schema, redis_client=self.client)
        try:
            self.index.create(overwrite=False)
        except Exception:
            # Index already initialized
            pass