"""
Search endpoints for document retrieval.
"""
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.dependencies import get_rag_service, get_current_user
from app.models.schemas import SearchResult, SearchFilters
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SearchType(str, Enum):
    """Search type enumeration."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(SearchType.HYBRID, description="Search type")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    limit: int = Field(10, ge=1, le=100, description="Number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    include_metadata: bool = Field(True, description="Include document metadata")


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[SearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")
    search_type: SearchType = Field(..., description="Search type used")
    execution_time: float = Field(..., description="Execution time in seconds")


@router.post("/", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """
    Search documents using various search strategies.
    
    Args:
        request: Search request
        rag_service: RAG service
        current_user: Current user (optional)
        
    Returns:
        SearchResponse: Search results
    """
    try:
        import time
        start_time = time.time()
        
        # Perform search
        results, total = await rag_service.search_documents(
            query=request.query,
            search_type=request.search_type.value,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            include_metadata=request.include_metadata,
            user_id=current_user.get("id") if current_user else None,
        )
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            total=total,
            query=request.query,
            search_type=request.search_type,
            execution_time=execution_time,
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching documents"
        )


@router.get("/suggestions", response_model=List[str])
async def get_search_suggestions(
    q: str = Query(..., description="Query prefix"),
    limit: int = Query(10, ge=1, le=20, description="Number of suggestions"),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Get search suggestions based on query prefix.
    
    Args:
        q: Query prefix
        limit: Number of suggestions
        rag_service: RAG service
        
    Returns:
        List[str]: Search suggestions
    """
    try:
        suggestions = await rag_service.get_search_suggestions(q, limit)
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while getting suggestions"
        )


@router.get("/filters", response_model=Dict[str, Any])
async def get_search_filters(
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Get available search filters.
    
    Args:
        rag_service: RAG service
        
    Returns:
        Dict: Available filters
    """
    try:
        filters = await rag_service.get_available_filters()
        return filters
        
    except Exception as e:
        logger.error(f"Error getting filters: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while getting filters"
        )


@router.get("/similar/{document_id}", response_model=List[SearchResult])
async def find_similar_documents(
    document_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of similar documents"),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Find documents similar to a given document.
    
    Args:
        document_id: Document ID
        limit: Number of similar documents
        rag_service: RAG service
        
    Returns:
        List[SearchResult]: Similar documents
    """
    try:
        similar_docs = await rag_service.find_similar_documents(document_id, limit)
        return similar_docs
        
    except Exception as e:
        logger.error(f"Error finding similar documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while finding similar documents"
        )


@router.get("/trending", response_model=List[str])
async def get_trending_queries(
    limit: int = Query(10, ge=1, le=20, description="Number of trending queries"),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Get trending search queries.
    
    Args:
        limit: Number of trending queries
        rag_service: RAG service
        
    Returns:
        List[str]: Trending queries
    """
    try:
        trending = await rag_service.get_trending_queries(limit)
        return trending
        
    except Exception as e:
        logger.error(f"Error getting trending queries: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while getting trending queries"
        )
