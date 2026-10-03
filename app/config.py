from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "Semantic LLM Cache"
    DEBUG: bool = False
    
    # Provider Settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key for embeddings and completions")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536
    
    # Redis / Vector Store Settings
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_INDEX_NAME: str = "llm_semantic_cache"
    CACHE_PREFIX: str = "cache:llm:"
    
    # Semantic Thresholds
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.95
    DEFAULT_CACHE_TTL_SECONDS: int = 86400  # 24 hours
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()