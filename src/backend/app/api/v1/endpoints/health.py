"""
Health check endpoints.
"""
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_database_service, get_redis_client, get_neo4j_client
from app.models.database import get_db_session
from app.services.database import DatabaseService
from app.services.redis_client import RedisClient
from app.services.neo4j_client import Neo4jClient
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Dict: Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "MOSDAC AI Knowledge Navigator",
        "version": "1.0.0",
    }


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    db_service: DatabaseService = Depends(get_database_service),
    redis_client: RedisClient = Depends(get_redis_client),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Detailed health check including database connections.
    
    Returns:
        Dict: Detailed health status information
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "MOSDAC AI Knowledge Navigator",
        "version": "1.0.0",
        "checks": {}
    }
    
    # Check PostgreSQL database
    try:
        await db_service.health_check()
        health_status["checks"]["postgresql"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        health_status["checks"]["postgresql"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        await redis_client.health_check()
        health_status["checks"]["redis"] = {
            "status": "healthy",
            "message": "Redis connection successful"
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check Neo4j
    try:
        await neo4j_client.health_check()
        health_status["checks"]["neo4j"] = {
            "status": "healthy",
            "message": "Neo4j connection successful"
        }
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        health_status["checks"]["neo4j"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@router.get("/metrics", response_model=Dict[str, Any])
async def health_metrics():
    """
    Get system metrics.
    
    Returns:
        Dict: System metrics
    """
    import psutil
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
        },
        "process": {
            "pid": psutil.Process().pid,
            "memory_info": psutil.Process().memory_info()._asdict(),
            "cpu_percent": psutil.Process().cpu_percent(),
            "num_threads": psutil.Process().num_threads(),
        }
    }
