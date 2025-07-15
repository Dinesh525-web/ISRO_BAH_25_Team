"""
Main ingestion pipeline orchestrator for MOSDAC data processing.
"""
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

from app.core.config import settings
from app.core.exceptions import ScrapingException
from app.data_ingestion.scrapers.mosdac_scraper import MOSDACscraper
from app.data_ingestion.scrapers.realtime_scraper import RealtimeScraper
from app.data_ingestion.processors.data_processor import DataProcessor
from app.data_ingestion.processors.text_processor import TextProcessor
from app.services.database import DatabaseService
from app.services.redis_client import RedisClient
from app.services.embedding_service import EmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class PipelineStep(str, Enum):
    """Pipeline execution steps."""
    SCRAPING = "scraping"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    INDEXING = "indexing"


class IngestionPipeline:
    """Main ingestion pipeline for MOSDAC data."""
    
    def __init__(self):
        self.status = PipelineStatus.IDLE
        self.current_step = None
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self.statistics = {
            'total_pages_scraped': 0,
            'total_pages_processed': 0,
            'total_documents_created': 0,
            'total_embeddings_generated': 0,
            'errors': [],
            'warnings': [],
        }
        self.callbacks = {}
        
    async def execute_full_pipeline(
        self,
        scraping_config: Dict[str, Any],
        processing_config: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete data ingestion pipeline.
        
        Args:
            scraping_config: Configuration for scraping
            processing_config: Configuration for processing
            user_id: User ID for tracking
            
        Returns:
            Pipeline execution results
        """
        try:
            self.status = PipelineStatus.RUNNING
            self.start_time = datetime.utcnow()
            self.progress = 0.0
            
            logger.info("Starting full ingestion pipeline")
            
            # Step 1: Web Scraping
            await self._execute_step(PipelineStep.SCRAPING)
            scraped_data = await self._scrape_data(scraping_config)
            self.statistics['total_pages_scraped'] = len(scraped_data)
            self.progress = 0.2
            
            # Step 2: Data Processing
            await self._execute_step(PipelineStep.PROCESSING)
            processed_data = await self._process_data(scraped_data, processing_config)
            self.statistics['total_pages_processed'] = len(processed_data)
            self.progress = 0.4
            
            # Step 3: Generate Embeddings
            await self._execute_step(PipelineStep.EMBEDDING)
            embedded_data = await self._generate_embeddings(processed_data)
            self.statistics['total_embeddings_generated'] = len(embedded_data)
            self.progress = 0.6
            
            # Step 4: Store in Database
            await self._execute_step(PipelineStep.STORAGE)
            stored_documents = await self._store_data(embedded_data, user_id)
            self.statistics['total_documents_created'] = len(stored_documents)
            self.progress = 0.8
            
            # Step 5: Index for Search
            await self._execute_step(PipelineStep.INDEXING)
            await self._index_documents(stored_documents)
            self.progress = 1.0
            
            # Pipeline completed successfully
            self.status = PipelineStatus.COMPLETED
            self.end_time = datetime.utcnow()
            
            logger.info("Full ingestion pipeline completed successfully")
            
            return {
                'status': self.status,
                'statistics': self.statistics,
                'execution_time': (self.end_time - self.start_time).total_seconds(),
                'documents_created': stored_documents,
            }
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.end_time = datetime.utcnow()
            self.statistics['errors'].append(str(e))
            
            logger.error(f"Pipeline execution failed: {e}")
            raise ScrapingException(f"Pipeline execution failed: {str(e)}")
    
    async def _execute_step(self, step: PipelineStep):
        """Execute a pipeline step with callbacks."""
        self.current_step = step
        logger.info(f"Executing pipeline step: {step}")
        
        # Execute pre-step callbacks
        if step in self.callbacks:
            for callback in self.callbacks[step]:
                try:
                    await callback('pre', step, self.statistics)
                except Exception as e:
                    logger.warning(f"Pre-step callback failed: {e}")
    
    async def _scrape_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape data from MOSDAC website."""
        scraped_data = []
        
        try:
            # Initialize scrapers
            scrapers = []
            
            if config.get('scrape_static', True):
                scrapers.append(MOSDACscraper())
            
            if config.get('scrape_realtime', False):
                scrapers.append(RealtimeScraper())
            
            # Run scrapers
            for scraper in scrapers:
                async with scraper:
                    if isinstance(scraper, MOSDACscraper):
                        if config.get('specific_sections'):
                            data = await scraper.scrape_specific_sections(
                                config['specific_sections']
                            )
                        else:
                            data = await scraper.scrape_all(
                                max_pages=config.get('max_pages', 100)
                            )
                    else:
                        data = await scraper.scrape_all(
                            max_pages=config.get('max_pages', 50)
                        )
                    
                    scraped_data.extend(data)
            
            logger.info(f"Scraped {len(scraped_data)} pages")
            return scraped_data
            
        except Exception as e:
            logger.error(f"Error in scraping phase: {e}")
            raise
    
    async def _process_data(
        self,
        scraped_data: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process scraped data."""
        try:
            # Initialize processors
            db_service = DatabaseService()
            redis_client = RedisClient()
            
            data_processor = DataProcessor(db_service, redis_client)
            text_processor = TextProcessor()
            
            # Process data in batches
            batch_size = config.get('batch_size', 10)
            processed_data = await data_processor.batch_process(
                scraped_data, batch_size
            )
            
            # Enhanced text processing
            for item in processed_data:
                if item.get('content'):
                    # Extract additional insights
                    item['readability'] = text_processor.calculate_readability(
                        item['content']
                    )
                    item['satellite_info'] = text_processor.extract_satellite_info(
                        item['content']
                    )
                    item['technical_terms'] = text_processor.extract_technical_terms(
                        item['content']
                    )
                    item['summary'] = text_processor.summarize_text(
                        item['content']
                    )
            
            logger.info(f"Processed {len(processed_data)} items")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in processing phase: {e}")
            raise
    
    async def _generate_embeddings(
        self,
        processed_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for processed data."""
        try:
            embedding_service = EmbeddingService()
            
            # Prepare texts for embedding
            texts = []
            for item in processed_data:
                # Combine title and content for embedding
                text = f"{item.get('title', '')} {item.get('content', '')}"
                texts.append(text)
            
            # Generate embeddings in batches
            embeddings = await embedding_service.generate_embeddings(texts)
            
            # Add embeddings to data
            for item, embedding in zip(processed_data, embeddings):
                item['embedding'] = embedding
                item['embedding_model'] = settings.EMBEDDING_MODEL
                item['embedding_dimension'] = settings.EMBEDDING_DIMENSION
            
            logger.info(f"Generated embeddings for {len(processed_data)} items")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            raise
    
    async def _store_data(
        self,
        embedded_data: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[str]:
        """Store processed data in database."""
        try:
            db_service = DatabaseService()
            stored_documents = []
            
            for item in embedded_data:
                # Create document record
                document_id = await db_service.create_document(
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    content_type='text/html',
                    content_hash=item.get('content_hash', ''),
                    source_url=item.get('url', ''),
                    source_type='scraped',
                    page_type=item.get('page_type', 'general'),
                    language=item.get('language', 'en'),
                    metadata=item.get('metadata', {}),
                    tags=item.get('keywords', []),
                    category=self._categorize_content(item),
                    quality_score=item.get('quality_score', 0.0),
                    uploaded_by_id=user_id,
                )
                
                # Store chunks with embeddings
                if item.get('content'):
                    chunks = self._create_chunks(item['content'])
                    
                    for i, chunk in enumerate(chunks):
                        chunk_embedding = await self._generate_chunk_embedding(
                            chunk, item.get('embedding', [])
                        )
                        
                        await db_service.create_document_chunk(
                            document_id=document_id,
                            chunk_index=i,
                            content=chunk,
                            embedding=chunk_embedding,
                            embedding_model=settings.EMBEDDING_MODEL,
                            embedding_dimension=settings.EMBEDDING_DIMENSION,
                        )
                
                stored_documents.append(document_id)
            
            logger.info(f"Stored {len(stored_documents)} documents")
            return stored_documents
            
        except Exception as e:
            logger.error(f"Error in storage phase: {e}")
            raise
    
    def _categorize_content(self, item: Dict[str, Any]) -> str:
        """Categorize content based on analysis."""
        page_type = item.get('page_type', 'general')
        
        if page_type == 'faq':
            return 'FAQ'
        elif page_type == 'product':
            return 'Product'
        elif page_type == 'satellite':
            return 'Satellite'
        elif page_type == 'documentation':
            return 'Documentation'
        elif page_type == 'news':
            return 'News'
        
        # Content-based categorization
        content = item.get('content', '').lower()
        
        if any(term in content for term in ['weather', 'meteorology', 'climate']):
            return 'Weather'
        elif any(term in content for term in ['ocean', 'sea', 'marine']):
            return 'Ocean'
        elif any(term in content for term in ['agriculture', 'crop', 'farming']):
            return 'Agriculture'
        elif any(term in content for term in ['disaster', 'emergency', 'cyclone']):
            return 'Disaster Management'
        
        return 'General'
    
    def _create_chunks(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Create text chunks for embedding."""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_chunk_embedding(
        self,
        chunk: str,
        document_embedding: List[float]
    ) -> List[float]:
        """Generate embedding for a chunk."""
        try:
            embedding_service = EmbeddingService()
            chunk_embedding = await embedding_service.generate_single_embedding(chunk)
            return chunk_embedding
        except Exception as e:
            logger.warning(f"Failed to generate chunk embedding: {e}")
            return document_embedding  # Fallback to document embedding
    
    async def _index_documents(self, document_ids: List[str]):
        """Index documents for search."""
        try:
            # This would typically update search indices
            # For now, we'll just log the operation
            logger.info(f"Indexed {len(document_ids)} documents for search")
            
            # Update Redis cache
            redis_client = RedisClient()
            await redis_client.set(
                'last_indexing_time',
                datetime.utcnow().isoformat(),
                ttl=None
            )
            
        except Exception as e:
            logger.error(f"Error in indexing phase: {e}")
            raise
    
    async def start_scraping_job(
        self,
        job_id: str,
        config: Dict[str, Any],
        user_id: str
    ):
        """Start a scraping job in the background."""
        try:
            logger.info(f"Starting scraping job {job_id}")
            
            # Update job status
            await self._update_job_status(job_id, PipelineStatus.RUNNING)
            
            # Execute pipeline
            result = await self.execute_full_pipeline(
                scraping_config=config,
                processing_config={},
                user_id=user_id
            )
            
            # Update job status
            await self._update_job_status(job_id, PipelineStatus.COMPLETED, result)
            
        except Exception as e:
            logger.error(f"Scraping job {job_id} failed: {e}")
            await self._update_job_status(job_id, PipelineStatus.FAILED, {'error': str(e)})
    
    async def _update_job_status(
        self,
        job_id: str,
        status: PipelineStatus,
        result: Optional[Dict[str, Any]] = None
    ):
        """Update job status in database."""
        try:
            redis_client = RedisClient()
            
            job_data = {
                'job_id': job_id,
                'status': status,
                'updated_at': datetime.utcnow().isoformat(),
                'progress': self.progress,
                'current_step': self.current_step,
                'statistics': self.statistics,
            }
            
            if result:
                job_data['result'] = result
            
            await redis_client.set(
                f"scraping_job:{job_id}",
                job_data,
                ttl=86400  # 24 hours
            )
            
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
    
    def add_callback(self, step: PipelineStep, callback: Callable):
        """Add a callback for a specific pipeline step."""
        if step not in self.callbacks:
            self.callbacks[step] = []
        self.callbacks[step].append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'status': self.status,
            'current_step': self.current_step,
            'progress': self.progress,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'statistics': self.statistics,
        }
    
    async def pause_pipeline(self):
        """Pause the pipeline execution."""
        self.status = PipelineStatus.PAUSED
        logger.info("Pipeline paused")
    
    async def resume_pipeline(self):
        """Resume the pipeline execution."""
        self.status = PipelineStatus.RUNNING
        logger.info("Pipeline resumed")
    
    async def cleanup(self):
        """Clean up pipeline resources."""
        try:
            # Clean up temporary files, connections, etc.
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
