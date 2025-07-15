"""
Logging configuration and utilities.
"""
import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import settings


def configure_logging():
    """Configure structured logging."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console" if settings.LOG_FORMAT != "json" else "json",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": settings.LOG_FILE,
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": settings.LOG_LEVEL,
                "propagate": True,
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        structlog.stdlib.BoundLogger: Configured logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__name__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                "Function executed successfully",
                function=func.__name__,
                execution_time=execution_time,
                args_count=len(args),
                kwargs_count=len(kwargs),
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Function execution failed",
                function=func.__name__,
                execution_time=execution_time,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__name__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                "Function executed successfully",
                function=func.__name__,
                execution_time=execution_time,
                args_count=len(args),
                kwargs_count=len(kwargs),
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Function execution failed",
                function=func.__name__,
                execution_time=execution_time,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def log_api_call(endpoint: str, method: str):
    """Decorator to log API calls."""
    def decorator(func):
        import time
        from functools import wraps
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger("api")
            start_time = time.time()
            
            # Extract request information
            request_id = kwargs.get('request_id', 'unknown')
            user_id = kwargs.get('user_id', 'anonymous')
            
            logger.info(
                "API call started",
                endpoint=endpoint,
                method=method,
                request_id=request_id,
                user_id=user_id,
            )
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    "API call completed",
                    endpoint=endpoint,
                    method=method,
                    request_id=request_id,
                    user_id=user_id,
                    execution_time=execution_time,
                    status="success",
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    "API call failed",
                    endpoint=endpoint,
                    method=method,
                    request_id=request_id,
                    user_id=user_id,
                    execution_time=execution_time,
                    status="error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                
                raise
        
        return wrapper
    return decorator


# Initialize logging on module import
configure_logging()
