from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings loaded from Environment Variables.
    No URLs, model names, or collection names are hardcoded.
    """
    # Server Settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    DEBUG: bool = True

    # Local AI Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_QDRANT_URL: str = "http://localhost:6333"
    LOCAL_LLM_MODEL: str = "llama3.1"
    LOCAL_EMBEDDING_MODEL: str = "qwen-embedding"
    LOCAL_COLLECTION_NAME: str = "my_documents"

    # Cloud AI Configuration
    GEMINI_API_KEY: str = ""
    QDRANT_CLOUD_URL: str = ""
    QDRANT_CLOUD_API_KEY: str = ""
    CLOUD_LLM_MODEL: str = "gemini-2.5-flash"
    CLOUD_EMBEDDING_MODEL: str = "text-embedding-004"
    CLOUD_COLLECTION_NAME: str = "my_documents_cloud"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    """
    return Settings()
