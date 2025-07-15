"""
Document management endpoints.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.dependencies import get_rag_service, get_current_user
from app.models.schemas import Document, DocumentMetadata, DocumentStatus
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[Document] = Field(..., description="Documents")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Limit")
    offset: int = Field(..., description="Offset")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    rag_service: RAGService = Depends(get_rag_service),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """
    Upload a document for processing.
    
    Args:
        background_tasks: Background tasks
        file: Upload file
        title: Document title
        description: Document description
        tags: Document tags (comma-separated)
        category: Document category
        rag_service: RAG service
        current_user: Current user
        
    Returns:
        DocumentUploadResponse: Upload response
    """
    try:
        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/csv",
            "application/json",
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Check file size (max 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Reset file position
        await file.seek(0)
        
        # Process document metadata
        metadata = DocumentMetadata(
            title=title or file.filename,
            description=description,
            tags=tags.split(",") if tags else [],
            category=category,
            content_type=file.content_type,
            file_size=len(content),
            uploaded_by=current_user.get("id") if current_user else None,
        )
        
        # Start document processing in background
        document_id = await rag_service.upload_document(
            file=file,
            metadata=metadata,
            background_tasks=background_tasks,
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status=DocumentStatus.PROCESSING,
            message="Document uploaded and processing started",
        )
        
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while uploading the document"
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    category: Optional[str] = None,
    status: Optional[DocumentStatus] = None,
    search: Optional[str] = None,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """
    List documents with optional filtering.
    
    Args:
        limit: Number of documents to return
        offset: Offset for pagination
        category: Filter by category
        status: Filter by status
        search: Search query
        rag_service: RAG service
        current_user: Current user
        
    Returns:
        DocumentListResponse: Document list
    """
    try:
        documents, total = await rag_service.list_documents(
            limit=limit,
            offset=offset,
            category=category,
            status=status,
            search=search,
            user_id=current_user.get("id") if current_user else None,
        )
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            limit=limit,
            offset=offset,
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while listing documents"
        )


@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """
    Get a specific document.
    
    Args:
        document_id: Document ID
        rag_service: RAG service
        current_user: Current user
        
    Returns:
        Document: Document details
    """
    try:
        document = await rag_service.get_document(
            document_id=document_id,
            user_id=current_user.get("id") if current_user else None,
        )
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        return document
        
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving the document"
        )


@router.put("/{document_id}", response_model=Document)
async def update_document(
    document_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: dict = Depends(get_current_user),
):
    """
    Update document metadata.
    
    Args:
        document_id: Document ID
        title: New title
        description: New description
        tags: New tags
        category: New category
        rag_service: RAG service
        current_user: Current user
        
    Returns:
        Document: Updated document
    """
    try:
        document = await rag_service.update_document(
            document_id=document_id,
            title=title,
            description=description,
            tags=tags,
            category=category,
            user_id=current_user["id"],
        )
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        return document
        
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while updating the document"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a document.
    
    Args:
        document_id: Document ID
        rag_service: RAG service
        current_user: Current user
        
    Returns:
        Dict: Success message
    """
    try:
        success = await rag_service.delete_document(
            document_id=document_id,
            user_id=current_user["id"],
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the document"
        )


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: dict = Depends(get_current_user),
):
    """
    Reprocess a document.
    
    Args:
        document_id: Document ID
        background_tasks: Background tasks
        rag_service: RAG service
        current_user: Current user
        
    Returns:
        Dict: Success message
    """
    try:
        success = await rag_service.reprocess_document(
            document_id=document_id,
            user_id=current_user["id"],
            background_tasks=background_tasks,
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        return {"message": "Document reprocessing started"}
        
    except Exception as e:
        logger.error(f"Error reprocessing document: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while reprocessing the document"
        )
