"""
Chat endpoints for conversational AI.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.dependencies import get_rag_service, get_current_user
from app.models.schemas import ChatMessage, ChatSession, ChatResponse
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[UUID] = Field(None, description="Chat session ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    stream: bool = Field(False, description="Enable streaming response")


class ChatStreamResponse(BaseModel):
    """Chat stream response model."""
    chunk: str = Field(..., description="Response chunk")
    session_id: UUID = Field(..., description="Chat session ID")
    message_id: UUID = Field(..., description="Message ID")
    finished: bool = Field(False, description="Stream finished")


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: Optional[dict] = Depends(get_current_user),
):
    """
    Chat with the AI assistant.
    
    Args:
        request: Chat request
        background_tasks: Background tasks
        rag_service: RAG service
        current_user: Current user (optional)
        
    Returns:
        ChatResponse: AI response
    """
    try:
        # Get or create session
        session_id = request.session_id or uuid4()
        
        # Process the message through RAG
        response = await rag_service.process_message(
            message=request.message,
            session_id=session_id,
            user_id=current_user.get("id") if current_user else None,
            context=request.context,
        )
        
        # Store conversation in background
        background_tasks.add_task(
            rag_service.store_conversation,
            session_id=session_id,
            user_message=request.message,
            ai_response=response.content,
            user_id=current_user.get("id") if current_user else None,
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your message"
        )


@router.get("/sessions", response_model=List[ChatSession])
async def get_chat_sessions(
    current_user: dict = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Get user's chat sessions.
    
    Args:
        current_user: Current authenticated user
        rag_service: RAG service
        
    Returns:
        List[ChatSession]: User's chat sessions
    """
    try:
        sessions = await rag_service.get_user_sessions(current_user["id"])
        return sessions
        
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving sessions"
        )


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: UUID,
    current_user: dict = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Get messages for a specific chat session.
    
    Args:
        session_id: Chat session ID
        current_user: Current authenticated user
        rag_service: RAG service
        
    Returns:
        List[ChatMessage]: Session messages
    """
    try:
        messages = await rag_service.get_session_messages(
            session_id=session_id,
            user_id=current_user["id"]
        )
        return messages
        
    except Exception as e:
        logger.error(f"Error getting session messages: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving messages"
        )


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: UUID,
    current_user: dict = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Delete a chat session.
    
    Args:
        session_id: Chat session ID
        current_user: Current authenticated user
        rag_service: RAG service
        
    Returns:
        Dict: Success message
    """
    try:
        await rag_service.delete_session(
            session_id=session_id,
            user_id=current_user["id"]
        )
        return {"message": "Session deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while deleting the session"
        )


@router.post("/feedback")
async def provide_feedback(
    message_id: UUID,
    rating: int = Field(..., ge=1, le=5),
    feedback: Optional[str] = None,
    current_user: Optional[dict] = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Provide feedback on AI response.
    
    Args:
        message_id: Message ID
        rating: Rating (1-5)
        feedback: Optional feedback text
        current_user: Current user
        rag_service: RAG service
        
    Returns:
        Dict: Success message
    """
    try:
        await rag_service.store_feedback(
            message_id=message_id,
            rating=rating,
            feedback=feedback,
            user_id=current_user.get("id") if current_user else None,
        )
        return {"message": "Feedback stored successfully"}
        
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while storing feedback"
        )
