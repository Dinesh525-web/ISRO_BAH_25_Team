"""
Custom exception classes for the application.
"""
from typing import Optional


class CustomException(Exception):
    """Base custom exception class."""
    
    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(detail)


class ValidationException(CustomException):
    """Exception for validation errors."""
    
    def __init__(self, detail: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(detail, 400, error_code)


class AuthenticationException(CustomException):
    """Exception for authentication errors."""
    
    def __init__(self, detail: str = "Authentication required", error_code: str = "AUTH_ERROR"):
        super().__init__(detail, 401, error_code)


class AuthorizationException(CustomException):
    """Exception for authorization errors."""
    
    def __init__(self, detail: str = "Insufficient permissions", error_code: str = "AUTHZ_ERROR"):
        super().__init__(detail, 403, error_code)


class NotFoundException(CustomException):
    """Exception for resource not found errors."""
    
    def __init__(self, detail: str = "Resource not found", error_code: str = "NOT_FOUND"):
        super().__init__(detail, 404, error_code)


class ConflictException(CustomException):
    """Exception for resource conflict errors."""
    
    def __init__(self, detail: str = "Resource conflict", error_code: str = "CONFLICT"):
        super().__init__(detail, 409, error_code)


class DatabaseException(CustomException):
    """Exception for database errors."""
    
    def __init__(self, detail: str = "Database error", error_code: str = "DB_ERROR"):
        super().__init__(detail, 500, error_code)


class ExternalServiceException(CustomException):
    """Exception for external service errors."""
    
    def __init__(self, detail: str = "External service error", error_code: str = "EXTERNAL_ERROR"):
        super().__init__(detail, 503, error_code)


class ScrapingException(CustomException):
    """Exception for web scraping errors."""
    
    def __init__(self, detail: str = "Scraping error", error_code: str = "SCRAPING_ERROR"):
        super().__init__(detail, 500, error_code)


class EmbeddingException(CustomException):
    """Exception for embedding generation errors."""
    
    def __init__(self, detail: str = "Embedding error", error_code: str = "EMBEDDING_ERROR"):
        super().__init__(detail, 500, error_code)


class RAGException(CustomException):
    """Exception for RAG system errors."""
    
    def __init__(self, detail: str = "RAG system error", error_code: str = "RAG_ERROR"):
        super().__init__(detail, 500, error_code)


class KnowledgeGraphException(CustomException):
    """Exception for knowledge graph errors."""
    
    def __init__(self, detail: str = "Knowledge graph error", error_code: str = "KG_ERROR"):
        super().__init__(detail, 500, error_code)
