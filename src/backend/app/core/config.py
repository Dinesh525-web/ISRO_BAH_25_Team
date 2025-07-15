"""
Application configuration settings.
"""
import os
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = Field(default="MOSDAC AI Knowledge Navigator")
    APP_VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)
    SECRET_KEY: str = Field(default="your-secret-key-here")
    ENVIRONMENT: str = Field(default="development")
    
    # API
    API_V1_STR: str = Field(default="/api/v1")
    BACKEND_HOST: str = Field(default="localhost")
    BACKEND_PORT: int = Field(default=8000)
    FRONTEND_HOST: str = Field(default="localhost")
    FRONTEND_PORT: int = Field(default=3000)
    
    # Database
    DATABASE_URL: str = Field(default="postgresql://postgres:password@localhost:5432/mosdac_ai")
    DATABASE_POOL_SIZE: int = Field(default=20)
    DATABASE_MAX_OVERFLOW: int = Field(default=30)
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_CACHE_TTL: int = Field(default=3600)
    
    # Neo4j
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")
    NEO4J_DATABASE: str = Field(default="mosdac")
    
    # OpenAI
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4-turbo-preview")
    OPENAI_TEMPERATURE: float = Field(default=0.1)
    OPENAI_MAX_TOKENS: int = Field(default=2000)
    
    # Pinecone (Optional)
    PINECONE_API_KEY: Optional[str] = Field(default=None)
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None)
    PINECONE_INDEX_NAME: Optional[str] = Field(default=None)
    
    # MOSDAC
    MOSDAC_BASE_URL: str = Field(default="https://www.mosdac.gov.in")
    MOSDAC_API_TIMEOUT: int = Field(default=30)
    SCRAPING_DELAY: int = Field(default=1)
    SCRAPING_RETRY_TIMES: int = Field(default=3)
    SCRAPING_CONCURRENT_REQUESTS: int = Field(default=8)
    
    # Embeddings
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")
    LOG_FILE: str = Field(default="logs/application.log")
    
    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    REFRESH_TOKEN_EXPIRE_MINUTES: int = Field(default=43200)
    ALGORITHM: str = Field(default="HS256")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=True)
    SENTRY_DSN: Optional[str] = Field(default=None)
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100)
    RATE_LIMIT_BURST: int = Field(default=10)
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=[
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
    ])
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from environment."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
