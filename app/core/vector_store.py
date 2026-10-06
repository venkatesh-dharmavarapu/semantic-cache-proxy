import logging
from redisvl.index import SearchIndex
from redisvl.redis.connection import RedisConnectionFactory
from app.config import settings
from app.core.schema import INDEX_SCHEMA

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self.index = SearchIndex.from_dict(INDEX_SCHEMA)
        self.client = None

    def connect(self):
        """Establish connection to Redis using recommended RedisVL connection factory."""
        self.client = RedisConnectionFactory.get_redis_connection(self.redis_url)
        self.index.set_client(self.client)

    def create_index(self, overwrite: bool = False):
        """Create the vector index in Redis if it does not already exist."""
        if not self.client:
            self.connect()
        try:
            self.index.create(overwrite=overwrite)
            logger.info("Search index '%s' ready.", settings.CACHE_INDEX_NAME)
        except Exception as e:
            logger.warning("Index creation notice: %s", e)

    def check_exists(self) -> bool:
        """Check if index exists in Redis."""
        if not self.client:
            self.connect()
        return self.index.exists()