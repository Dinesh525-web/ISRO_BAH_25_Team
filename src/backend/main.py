"""
MOSDAC AI Knowledge Navigator - Main FastAPI Application
========================================================

This is the main entry point for the MOSDAC AI Knowledge Navigator backend.
It sets up the FastAPI application with all necessary middleware, routes, and configurations.

Team: GravitasOps
- Dinesh Yadav (Team Leader)
- Sachin Munjar (Backend Developer)
- Anushka Jain (Frontend Developer)  
- Aditya Raj (DevOps/Data Engineer)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging import setup_logging
from app.core.dependencies import get_current_user
from app.api.v1.endpoints import chat, health, data_ingestion, knowledge_graph, analytics
from app.middleware.logging import LoggingMiddleware
from app.middleware.auth import AuthMiddleware
from app.services.rag_service import RAGService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.vector_service import VectorService

# =============================================================================
# METRICS SETUP
# =============================================================================
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

RAG_QUERY_COUNT = Counter(
    'rag_queries_total',
    'Total RAG queries processed',
    ['query_type', 'status']
)

RAG_RESPONSE_TIME = Histogram(
    'rag_response_time_seconds',
    'RAG response time in seconds',
    ['query_type']
)

# =============================================================================
# LOGGING SETUP
# =============================================================================
setup_logging()
logger = structlog.get_logger(__name__)

# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting MOSDAC AI Knowledge Navigator")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        # Note: In production, use Alembic for migrations
        # Base.metadata.create_all(bind=engine)
        
        # Initialize AI services
        logger.info("Initializing AI services...")
        rag_service = RAGService()
        kg_service = KnowledgeGraphService()
        vector_service = VectorService()
        
        # Warm up models
        logger.info("Warming up AI models...")
        await rag_service.initialize()
        await kg_service.initialize()
        await vector_service.initialize()
        
        # Store services in app state
        app.state.rag_service = rag_service
        app.state.kg_service = kg_service
        app.state.vector_service = vector_service
        
        logger.info("MOSDAC AI Knowledge Navigator started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down MOSDAC AI Knowledge Navigator")
        
        # Cleanup services
        if hasattr(app.state, 'rag_service'):
            await app.state.rag_service.cleanup()
        if hasattr(app.state, 'kg_service'):
            await app.state.kg_service.cleanup()
        if hasattr(app.state, 'vector_service'):
            await app.state.vector_service.cleanup()
        
        logger.info("MOSDAC AI Knowledge Navigator shut down successfully")

# =============================================================================
# MIDDLEWARE CLASSES
# =============================================================================
class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting Prometheus metrics.
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract endpoint and method
        method = request.method
        endpoint = request.url.path
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            logger.warning(f"HTTP exception: {e.detail}", status_code=e.status_code)
            raise
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )

# =============================================================================
# APPLICATION SETUP
# =============================================================================
app = FastAPI(
    title="MOSDAC AI Knowledge Navigator",
    description="""
    AI-powered conversational assistant for MOSDAC satellite data portal.
    
    This API provides intelligent responses to user queries about meteorological
    and oceanographic satellite data using RAG (Retrieval-Augmented Generation)
    and knowledge graph technologies.
    
    **Key Features:**
    - Natural language query processing
    - Knowledge graph-based information retrieval
    - Real-time data updates
    - Multilingual support
    - Interactive data visualization
    
    **Team:** GravitasOps for ISRO Bharatiya Antariksh Hackathon 2025
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_REDOC else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

# =============================================================================
# MIDDLEWARE SETUP
# =============================================================================
# Error handling (first middleware)
app.add_middleware(ErrorHandlingMiddleware)

# Metrics collection
app.add_middleware(MetricsMiddleware)

# Request logging
app.add_middleware(LoggingMiddleware)

# Authentication (for protected routes)
app.add_middleware(AuthMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# =============================================================================
# STATIC FILES
# =============================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# API ROUTES
# =============================================================================
# Health check endpoint
app.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

# API v1 routes
app.include_router(
    chat.router,
    prefix=f"{settings.API_PREFIX}/chat",
    tags=["chat"],
    dependencies=[]  # Add authentication dependency if needed
)

app.include_router(
    data_ingestion.router,
    prefix=f"{settings.API_PREFIX}/data",
    tags=["data_ingestion"],
    dependencies=[]  # Add authentication dependency if needed
)

app.include_router(
    knowledge_graph.router,
    prefix=f"{settings.API_PREFIX}/knowledge-graph",
    tags=["knowledge_graph"],
    dependencies=[]  # Add authentication dependency if needed
)

app.include_router(
    analytics.router,
    prefix=f"{settings.API_PREFIX}/analytics",
    tags=["analytics"],
    dependencies=[]  # Add authentication dependency if needed
)

# =============================================================================
# METRICS ENDPOINT
# =============================================================================
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# =============================================================================
# ROOT ENDPOINT
# =============================================================================
@app.get("/")
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "message": "MOSDAC AI Knowledge Navigator API",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "team": "GravitasOps",
        "docs_url": "/docs" if settings.ENABLE_DOCS else None,
        "health_check": "/health",
        "api_prefix": settings.API_PREFIX
    }

# =============================================================================
# CUSTOM EXCEPTION HANDLERS
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler.
    """
    logger.warning(
        f"HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    General exception handler for unhandled exceptions.
    """
    logger.error(
        f"Unhandled exception occurred",
        exception=str(exc),
        path=request.url.path,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": time.time(),
            "path": request.url.path
        }
    )

# =============================================================================
# STARTUP MESSAGE
# =============================================================================
@app.on_event("startup")
async def startup_message():
    """
    Display startup message with configuration info.
    """
    logger.info(
        "MOSDAC AI Knowledge Navigator API starting up",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        debug=settings.DEBUG,
        database_url=settings.DATABASE_URL.replace(
            settings.DATABASE_PASSWORD, "***"
        ) if settings.DATABASE_URL else "Not configured"
    )

# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================
if settings.DEBUG:
    @app.get("/debug/config")
    async def debug_config():
        """
        Debug endpoint to view configuration (development only).
        """
        return {
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "app_env": settings.APP_ENV,
            "debug": settings.DEBUG,
            "api_prefix": settings.API_PREFIX,
            "cors_origins": settings.CORS_ORIGINS,
            "database_connected": bool(settings.DATABASE_URL),
            "redis_connected": bool(settings.REDIS_URL),
            "neo4j_connected": bool(settings.NEO4J_URI),
        }

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        use_colors=True,
        loop="asyncio"
    )