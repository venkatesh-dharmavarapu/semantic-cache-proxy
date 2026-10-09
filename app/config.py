from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    PROJECT_NAME: str = "Semantic LLM Cache (Ollama Powered)"
    DEBUG: bool = False

    # Ollama Provider Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIM: int = 768  # nomic-embed-text outputs 768-dim vectors
    DEFAULT_LLM_MODEL: str = "llama3.2:1b"

    # Redis / Vector Store Settings
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_INDEX_NAME: str = "llm_semantic_cache"
    CACHE_PREFIX: str = "cache:llm:"

    # Semantic Thresholds
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.88
    DEFAULT_CACHE_TTL_SECONDS: int = 86400  # 24 hours

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()