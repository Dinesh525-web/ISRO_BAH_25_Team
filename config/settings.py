"""
Application settings and configuration.
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = Field("MOSDAC AI Knowledge Navigator", env="APP_NAME")
    APP_VERSION: str = Field("1.0.0", env="APP_VERSION")
    DEBUG: bool = Field(False, env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ENVIRONMENT: str = Field("production", env="ENVIRONMENT")
    
    # API
    API_V1_STR: str = Field("/api/v1", env="API_V1_STR")
    BACKEND_HOST: str = Field("localhost", env="BACKEND_HOST")
    BACKEND_PORT: int = Field(8000, env="BACKEND_PORT")
    FRONTEND_HOST: str = Field("localhost", env="FRONTEND_HOST")
    FRONTEND_PORT: int = Field(3000, env="FRONTEND_PORT")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field([
        "http://localhost:3000",
        "http://localhost:8501",
        "https://mosdac.gov.in"
    ], env="ALLOWED_ORIGINS")
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    REDIS_CACHE_TTL: int = Field(3600, env="REDIS_CACHE_TTL")
    
    # Neo4j
    NEO4J_URI: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USERNAME: str = Field("neo4j", env="NEO4J_USERNAME")
    NEO4J_PASSWORD: str = Field(..., env="NEO4J_PASSWORD")
    NEO4J_DATABASE: str = Field("mosdac", env="NEO4J_DATABASE")
    
    # OpenAI
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")
    OPENAI_TEMPERATURE: float = Field(0.1, env="OPENAI_TEMPERATURE")
    OPENAI_MAX_TOKENS: int = Field(2000, env="OPENAI_MAX_TOKENS")
    
    # Pinecone (optional)
    PINECONE_API_KEY: Optional[str] = Field(None, env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: Optional[str] = Field(None, env="PINECONE_INDEX_NAME")
    
    # MOSDAC
    MOSDAC_BASE_URL: str = Field("https://www.mosdac.gov.in", env="MOSDAC_BASE_URL")
    MOSDAC_API_TIMEOUT: int = Field(30, env="MOSDAC_API_TIMEOUT")
    SCRAPING_DELAY: float = Field(1.0, env="SCRAPING_DELAY")
    SCRAPING_RETRY_TIMES: int = Field(3, env="SCRAPING_RETRY_TIMES")
    SCRAPING_CONCURRENT_REQUESTS: int = Field(8, env="SCRAPING_CONCURRENT_REQUESTS")
    
    # Embeddings
    EMBEDDING_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(384, env="EMBEDDING_DIMENSION")
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env="CHUNK_OVERLAP")
    
    # Logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field("json", env="LOG_FORMAT")
    LOG_FILE: str = Field("logs/application.log", env="LOG_FILE")
    
    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_MINUTES: int = Field(43200, env="REFRESH_TOKEN_EXPIRE_MINUTES")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(10, env="RATE_LIMIT_BURST")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(True, env="PROMETHEUS_ENABLED")
    SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
