"""
FastAPI dependency injection utilities.
"""
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db_session
from app.services.database import DatabaseService
from app.services.redis_client import RedisClient
from app.services.neo4j_client import Neo4jClient
from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


async def get_database_service(
    db: AsyncSession = Depends(get_db_session)
) -> DatabaseService:
    """Get database service instance."""
    return DatabaseService(db)


async def get_redis_client() -> RedisClient:
    """Get Redis client instance."""
    return RedisClient()


async def get_neo4j_client() -> Neo4jClient:
    """Get Neo4j client instance."""
    return Neo4jClient()


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService()


async def get_rag_service(
    db_service: DatabaseService = Depends(get_database_service),
    redis_client: RedisClient = Depends(get_redis_client),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RAGService:
    """Get RAG service instance."""
    return RAGService(
        db_service=db_service,
        redis_client=redis_client,
        neo4j_client=neo4j_client,
        embedding_service=embedding_service,
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Get current authenticated user (optional)."""
    if credentials is None:
        return None
    
    try:
        # TODO: Implement JWT token validation
        # For now, we'll return a mock user
        return {"id": "user123", "email": "user@example.com"}
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_authentication(
    current_user: Optional[dict] = Depends(get_current_user)
) -> dict:
    """Require authentication."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user
