"""
CORS middleware configuration for cross-origin requests.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def setup_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=[
                "Accept",
                "Accept-Language",
                "Content-Language",
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "X-API-Key",
                "X-Client-Version",
                "X-Request-ID",
                "Cache-Control",
                "Pragma",
                "Expires",
            ],
            expose_headers=[
                "X-Total-Count",
                "X-Page-Count",
                "X-Per-Page",
                "X-Current-Page",
                "X-Rate-Limit-Remaining",
                "X-Rate-Limit-Reset",
                "X-Response-Time",
            ],
            max_age=86400,  # 24 hours
        )
        
        logger.info("CORS middleware configured successfully")
        
    except Exception as e:
        logger.error(f"Error configuring CORS middleware: {e}")
        raise
