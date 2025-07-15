"""
Rate limiting middleware to prevent abuse.
"""
import time
from typing import Dict, Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import settings
from app.services.redis_client import RedisClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis."""
    
    def __init__(self, app, redis_client: Optional[RedisClient] = None):
        super().__init__(app)
        self.redis_client = redis_client or RedisClient()
        self.default_limit = settings.RATE_LIMIT_PER_MINUTE
        self.burst_limit = settings.RATE_LIMIT_BURST
        self.window_size = 60  # 1 minute window
        
        # Different limits for different endpoints
        self.endpoint_limits = {
            "/api/v1/chat": 20,  # Lower limit for chat endpoints
            "/api/v1/search": 30,  # Moderate limit for search
            "/api/v1/documents/upload": 5,  # Very low limit for uploads
            "/api/v1/scraper/start": 2,  # Very low limit for scraping
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Apply rate limiting to requests.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limit for this endpoint
        limit = self._get_endpoint_limit(request.url.path)
        
        # Check rate limit
        try:
            allowed, remaining, reset_time = await self._check_rate_limit(
                client_id, limit, request.url.path
            )
            
            if not allowed:
                # Rate limit exceeded
                logger.warning(
                    "Rate limit exceeded",
                    client_id=client_id,
                    endpoint=request.url.path,
                    limit=limit,
                )
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={
                        "X-Rate-Limit-Limit": str(limit),
                        "X-Rate-Limit-Remaining": str(remaining),
                        "X-Rate-Limit-Reset": str(reset_time),
                        "Retry-After": str(reset_time - int(time.time())),
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-Rate-Limit-Limit"] = str(limit)
            response.headers["X-Rate-Limit-Remaining"] = str(remaining)
            response.headers["X-Rate-Limit-Reset"] = str(reset_time)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in rate limiting: {e}")
            # Continue without rate limiting if Redis is down
            return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from request
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        # Fallback to IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_endpoint_limit(self, path: str) -> int:
        """Get rate limit for specific endpoint."""
        for endpoint, limit in self.endpoint_limits.items():
            if path.startswith(endpoint):
                return limit
        
        return self.default_limit
    
    async def _check_rate_limit(
        self,
        client_id: str,
        limit: int,
        endpoint: str
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Args:
            client_id: Client identifier
            limit: Rate limit
            endpoint: Endpoint path
            
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        try:
            current_time = int(time.time())
            window_start = current_time - (current_time % self.window_size)
            
            # Redis key for this client and time window
            key = f"rate_limit:{client_id}:{endpoint}:{window_start}"
            
            # Get current count
            current_count = await self.redis_client.get(key)
            current_count = int(current_count) if current_count else 0
            
            # Check if limit exceeded
            if current_count >= limit:
                reset_time = window_start + self.window_size
                return False, 0, reset_time
            
            # Increment counter
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, self.window_size)
            
            remaining = limit - current_count - 1
            reset_time = window_start + self.window_size
            
            return True, remaining, reset_time
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Allow request if Redis is unavailable
            return True, limit, current_time + self.window_size
