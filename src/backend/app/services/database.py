"""
Database service for ORM operations and queries.
"""
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update, delete, func, text, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.models.entities import (
    User, Document, DocumentChunk, ChatSession, ChatMessage,
    SearchQuery, Feedback, ScrapingJob, KnowledgeGraphNode,
    KnowledgeGraphRelation, Base
)
from app.models.schemas import DocumentStatus, SearchFilters
from app.core.exceptions import DatabaseException, NotFoundException
from app.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar('T', bound=Base)


class DatabaseService(Generic[T]):
    """Generic database service for ORM operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, model_class: Type[T], **kwargs) -> T:
        """
        Create a new record.
        
        Args:
            model_class: SQLAlchemy model class
            **kwargs: Model field values
            
        Returns:
            Created model instance
        """
        try:
            instance = model_class(**kwargs)
            self.db.add(instance)
            await self.db.commit()
            await self.db.refresh(instance)
            
            logger.debug(f"Created {model_class.__name__} with ID: {instance.id}")
            return instance
            
        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Integrity error creating {model_class.__name__}: {e}")
            raise DatabaseException(f"Record already exists or constraint violation: {str(e)}")
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error creating {model_class.__name__}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def get_by_id(self, model_class: Type[T], record_id: str) -> Optional[T]:
        """
        Get record by ID.
        
        Args:
            model_class: SQLAlchemy model class
            record_id: Record ID
            
        Returns:
            Model instance or None
        """
        try:
            stmt = select(model_class).where(model_class.id == record_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting {model_class.__name__} by ID {record_id}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def get_by_field(
        self,
        model_class: Type[T],
        field_name: str,
        field_value: Any
    ) -> Optional[T]:
        """
        Get record by field value.
        
        Args:
            model_class: SQLAlchemy model class
            field_name: Field name
            field_value: Field value
            
        Returns:
            Model instance or None
        """
        try:
            field = getattr(model_class, field_name)
            stmt = select(model_class).where(field == field_value)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting {model_class.__name__} by {field_name}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def list_records(
        self,
        model_class: Type[T],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        desc: bool = False
    ) -> List[T]:
        """
        List records with filtering and pagination.
        
        Args:
            model_class: SQLAlchemy model class
            filters: Field filters
            limit: Maximum records to return
            offset: Records to skip
            order_by: Field to order by
            desc: Descending order
            
        Returns:
            List of model instances
        """
        try:
            stmt = select(model_class)
            
            # Apply filters
            if filters:
                for field_name, field_value in filters.items():
                    if hasattr(model_class, field_name):
                        field = getattr(model_class, field_name)
                        if isinstance(field_value, list):
                            stmt = stmt.where(field.in_(field_value))
                        else:
                            stmt = stmt.where(field == field_value)
            
            # Apply ordering
            if order_by and hasattr(model_class, order_by):
                order_field = getattr(model_class, order_by)
                if desc:
                    order_field = order_field.desc()
                stmt = stmt.order_by(order_field)
            else:
                # Default order by created_at if available
                if hasattr(model_class, 'created_at'):
                    stmt = stmt.order_by(model_class.created_at.desc())
            
            # Apply pagination
            stmt = stmt.offset(offset).limit(limit)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error listing {model_class.__name__}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def update(
        self,
        model_class: Type[T],
        record_id: str,
        **kwargs
    ) -> Optional[T]:
        """
        Update record by ID.
        
        Args:
            model_class: SQLAlchemy model class
            record_id: Record ID
            **kwargs: Fields to update
            
        Returns:
            Updated model instance or None
        """
        try:
            # Add updated_at timestamp if field exists
            if hasattr(model_class, 'updated_at'):
                kwargs['updated_at'] = datetime.utcnow()
            
            stmt = (
                update(model_class)
                .where(model_class.id == record_id)
                .values(**kwargs)
                .returning(model_class)
            )
            
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            updated_record = result.scalar_one_or_none()
            if updated_record:
                await self.db.refresh(updated_record)
                logger.debug(f"Updated {model_class.__name__} with ID: {record_id}")
            
            return updated_record
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error updating {model_class.__name__} {record_id}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def delete(self, model_class: Type[T], record_id: str) -> bool:
        """
        Delete record by ID.
        
        Args:
            model_class: SQLAlchemy model class
            record_id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            stmt = delete(model_class).where(model_class.id == record_id)
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            deleted = result.rowcount > 0
            if deleted:
                logger.debug(f"Deleted {model_class.__name__} with ID: {record_id}")
            
            return deleted
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error deleting {model_class.__name__} {record_id}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def count(
        self,
        model_class: Type[T],
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count records with optional filters.
        
        Args:
            model_class: SQLAlchemy model class
            filters: Field filters
            
        Returns:
            Record count
        """
        try:
            stmt = select(func.count(model_class.id))
            
            # Apply filters
            if filters:
                for field_name, field_value in filters.items():
                    if hasattr(model_class, field_name):
                        field = getattr(model_class, field_name)
                        stmt = stmt.where(field == field_value)
            
            result = await self.db.execute(stmt)
            return result.scalar()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error counting {model_class.__name__}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    # Specialized methods for specific models
    
    async def create_document(self, **kwargs) -> Document:
        """Create a new document."""
        return await self.create(Document, **kwargs)
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID with chunks."""
        try:
            stmt = (
                select(Document)
                .options(selectinload(Document.chunks))
                .where(Document.id == document_id)
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def search_documents(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Document]:
        """Search documents by content."""
        try:
            stmt = select(Document).where(
                or_(
                    Document.title.ilike(f"%{query}%"),
                    Document.content.ilike(f"%{query}%")
                )
            )
            
            # Apply filters
            if filters:
                if filters.categories:
                    stmt = stmt.where(Document.category.in_(filters.categories))
                if filters.content_types:
                    stmt = stmt.where(Document.content_type.in_(filters.content_types))
                if filters.date_from:
                    stmt = stmt.where(Document.created_at >= filters.date_from)
                if filters.date_to:
                    stmt = stmt.where(Document.created_at <= filters.date_to)
                if filters.tags:
                    for tag in filters.tags:
                        stmt = stmt.where(Document.tags.contains([tag]))
                if filters.quality_score_min:
                    stmt = stmt.where(Document.quality_score >= filters.quality_score_min)
                if filters.language:
                    stmt = stmt.where(Document.language == filters.language)
            
            stmt = stmt.order_by(Document.created_at.desc()).offset(offset).limit(limit)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error searching documents: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def create_document_chunk(self, **kwargs) -> DocumentChunk:
        """Create a document chunk."""
        return await self.create(DocumentChunk, **kwargs)
    
    async def get_document_chunks(
        self,
        document_id: str,
        include_embeddings: bool = False
    ) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        try:
            stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            
            if not include_embeddings:
                # Exclude embedding field for performance
                stmt = stmt.options(selectinload(DocumentChunk).load_only(
                    DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.chunk_index,
                    DocumentChunk.content, DocumentChunk.content_hash, DocumentChunk.token_count
                ))
            
            stmt = stmt.order_by(DocumentChunk.chunk_index)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting document chunks: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def create_chat_session(self, **kwargs) -> ChatSession:
        """Create a chat session."""
        return await self.create(ChatSession, **kwargs)
    
    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID."""
        return await self.get_by_id(ChatSession, session_id)
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[ChatSession]:
        """Get user's chat sessions."""
        return await self.list_records(
            ChatSession,
            filters={"user_id": user_id, "is_active": True},
            limit=limit,
            order_by="last_activity",
            desc=True
        )
    
    async def create_chat_message(self, **kwargs) -> ChatMessage:
        """Create a chat message."""
        return await self.create(ChatMessage, **kwargs)
    
    async def get_session_messages(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[ChatMessage]:
        """Get messages for a chat session."""
        return await self.list_records(
            ChatMessage,
            filters={"session_id": session_id},
            limit=limit,
            order_by="created_at",
            desc=False
        )
    
    async def create_search_query(self, **kwargs) -> SearchQuery:
        """Create a search query record."""
        return await self.create(SearchQuery, **kwargs)
    
    async def get_trending_queries(self, limit: int = 10) -> List[str]:
        """Get trending search queries."""
        try:
            # Get most frequent queries from last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            stmt = (
                select(SearchQuery.query, func.count(SearchQuery.query).label('count'))
                .where(SearchQuery.created_at >= thirty_days_ago)
                .group_by(SearchQuery.query)
                .order_by(func.count(SearchQuery.query).desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            return [row[0] for row in result.all()]
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting trending queries: {e}")
            return []
    
    async def create_feedback(self, **kwargs) -> Feedback:
        """Create feedback record."""
        return await self.create(Feedback, **kwargs)
    
    async def create_knowledge_graph_node(self, **kwargs) -> KnowledgeGraphNode:
        """Create knowledge graph node."""
        return await self.create(KnowledgeGraphNode, **kwargs)
    
    async def create_knowledge_graph_relation(self, **kwargs) -> KnowledgeGraphRelation:
        """Create knowledge graph relation."""
        return await self.create(KnowledgeGraphRelation, **kwargs)
    
    async def get_node_relationships(
        self,
        node_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[KnowledgeGraphRelation]:
        """Get relationships for a node."""
        try:
            stmt = select(KnowledgeGraphRelation).where(
                or_(
                    KnowledgeGraphRelation.source_node_id == node_id,
                    KnowledgeGraphRelation.target_node_id == node_id
                )
            )
            
            if relation_types:
                stmt = stmt.where(KnowledgeGraphRelation.relation_type.in_(relation_types))
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting node relationships: {e}")
            raise DatabaseException(f"Database error: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            # Simple query to test connection
            result = await self.db.execute(text("SELECT 1"))
            return result.scalar() == 1
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Count records in main tables
            for model_class in [Document, ChatSession, ChatMessage, SearchQuery]:
                count = await self.count(model_class)
                stats[f"{model_class.__name__.lower()}_count"] = count
            
            # Document status distribution
            stmt = (
                select(Document.status, func.count(Document.id))
                .group_by(Document.status)
            )
            result = await self.db.execute(stmt)
            stats['document_status_distribution'] = dict(result.all())
            
            # Recent activity (last 24 hours)
            twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
            
            recent_documents = await self.count(
                Document,
                filters={"created_at": twenty_four_hours_ago}
            )
            stats['recent_documents'] = recent_documents
            
            recent_searches = await self.count(
                SearchQuery,
                filters={"created_at": twenty_four_hours_ago}
            )
            stats['recent_searches'] = recent_searches
            
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
