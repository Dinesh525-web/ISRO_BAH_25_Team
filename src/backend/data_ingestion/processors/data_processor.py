"""
Main data processor for scraped content processing and normalization.
"""
import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urljoin, urlparse

from .text_processor import TextProcessor
from app.services.database import DatabaseService
from app.services.redis_client import RedisClient
from app.core.exceptions import ScrapingException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """Main data processor for scraped content."""
    
    def __init__(self, db_service: DatabaseService, redis_client: RedisClient):
        self.db_service = db_service
        self.redis_client = redis_client
        self.text_processor = TextProcessor()
        
        # Quality thresholds
        self.min_content_length = 50
        self.min_quality_score = 0.3
        self.max_content_length = 100000
        
        # Content filters
        self.excluded_extensions = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.tar', '.gz', '.exe', '.dmg'
        ]
        
        self.excluded_content_types = [
            'application/pdf', 'application/msword', 'application/excel',
            'application/zip', 'application/x-zip-compressed'
        ]
    
    async def process_scraped_data(
        self,
        scraped_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process scraped data through full pipeline.
        
        Args:
            scraped_data: Raw scraped data
            
        Returns:
            Processed data
        """
        try:
            logger.info(f"Processing {len(scraped_data)} scraped items")
            
            processed_data = []
            
            for item in scraped_data:
                try:
                    # Validate item
                    if not self._validate_item(item):
                        continue
                    
                    # Normalize item
                    normalized_item = await self._normalize_item(item)
                    
                    # Extract content
                    content = await self._extract_content(normalized_item)
                    
                    # Process text
                    processed_content = await self._process_text_content(content)
                    
                    # Merge with original item
                    processed_item = {**normalized_item, **processed_content}
                    
                    # Calculate quality score
                    quality_score = self._calculate_quality_score(processed_item)
                    processed_item['quality_score'] = quality_score
                    
                    # Filter by quality
                    if quality_score >= self.min_quality_score:
                        processed_data.append(processed_item)
                        logger.debug(f"Processed item: {processed_item.get('title', 'Unknown')}")
                    else:
                        logger.debug(f"Filtered low quality item: {quality_score}")
                
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(processed_data)} items")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in process_scraped_data: {e}")
            raise ScrapingException(f"Data processing failed: {str(e)}")
    
    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate scraped item."""
        # Check required fields
        required_fields = ['url', 'content']
        for field in required_fields:
            if field not in item or not item[field]:
                logger.debug(f"Missing required field: {field}")
                return False
        
        # Check URL validity
        url = item.get('url', '')
        if not self._is_valid_url(url):
            logger.debug(f"Invalid URL: {url}")
            return False
        
        # Check content length
        content = item.get('content', '')
        if len(content) < self.min_content_length:
            logger.debug(f"Content too short: {len(content)} chars")
            return False
        
        if len(content) > self.max_content_length:
            logger.debug(f"Content too long: {len(content)} chars")
            return False
        
        # Check for excluded file types
        if self._is_excluded_content(item):
            logger.debug(f"Excluded content type: {item.get('content_type', '')}")
            return False
        
        return True
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        except:
            return False
    
    def _is_excluded_content(self, item: Dict[str, Any]) -> bool:
        """Check if content should be excluded."""
        # Check URL extension
        url = item.get('url', '').lower()
        if any(url.endswith(ext) for ext in self.excluded_extensions):
            return True
        
        # Check content type
        content_type = item.get('content_type', '').lower()
        if any(ct in content_type for ct in self.excluded_content_types):
            return True
        
        return False
    
    async def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize scraped item structure."""
        normalized = {
            'url': item.get('url', ''),
            'title': item.get('title', ''),
            'content': item.get('content', ''),
            'content_type': item.get('content_type', 'text/html'),
            'page_type': item.get('page_type', 'general'),
            'scraped_at': item.get('scraped_at', datetime.utcnow().isoformat()),
            'source_type': 'scraped',
            'language': 'en',
            'metadata': item.get('metadata', {}),
            'links': item.get('links', []),
            'images': item.get('images', []),
            'tables': item.get('tables', []),
            'forms': item.get('forms', []),
        }
        
        # Generate content hash
        content_hash = hashlib.md5(
            normalized['content'].encode('utf-8')
        ).hexdigest()
        normalized['content_hash'] = content_hash
        
        # Normalize URL
        normalized['url'] = self._normalize_url(normalized['url'])
        
        # Extract domain
        parsed_url = urlparse(normalized['url'])
        normalized['domain'] = parsed_url.netloc
        
        return normalized
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL format."""
        try:
            parsed = urlparse(url)
            
            # Remove fragment
            normalized = parsed._replace(fragment='').geturl()
            
            # Remove trailing slash
            if normalized.endswith('/') and normalized.count('/') > 2:
                normalized = normalized.rstrip('/')
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing URL: {e}")
            return url
    
    async def _extract_content(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure content from item."""
        content_data = {
            'raw_content': item.get('content', ''),
            'clean_content': '',
            'title': item.get('title', ''),
            'description': '',
            'main_content': '',
            'structured_data': {},
        }
        
        # Clean content
        raw_content = content_data['raw_content']
        clean_content = self.text_processor.clean_text(raw_content)
        content_data['clean_content'] = clean_content
        
        # Extract main content (remove navigation, footer, etc.)
        main_content = self._extract_main_content(clean_content, item)
        content_data['main_content'] = main_content
        
        # Extract description
        description = self._extract_description(item)
        content_data['description'] = description
        
        # Extract structured data
        structured_data = self._extract_structured_data(item)
        content_data['structured_data'] = structured_data
        
        return content_data
    
    def _extract_main_content(
        self,
        content: str,
        item: Dict[str, Any]
    ) -> str:
        """Extract main content, removing navigation and boilerplate."""
        # For now, use the cleaned content as main content
        # In a more sophisticated implementation, this would remove
        # navigation, footer, sidebar, and other boilerplate content
        
        # Remove common boilerplate phrases
        boilerplate_patterns = [
            r'copyright.*\d{4}',
            r'all rights reserved',
            r'privacy policy',
            r'terms of service',
            r'contact us',
            r'home\s*\|\s*about\s*\|\s*services',
            r'navigation menu',
            r'skip to main content',
            r'back to top',
        ]
        
        main_content = content
        for pattern in boilerplate_patterns:
            main_content = re.sub(pattern, '', main_content, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        main_content = re.sub(r'\s+', ' ', main_content).strip()
        
        return main_content
    
    def _extract_description(self, item: Dict[str, Any]) -> str:
        """Extract or generate description."""
        # Try to get description from metadata
        description = item.get('metadata', {}).get('description', '')
        
        if description:
            return description
        
        # Generate description from content
        content = item.get('content', '')
        if content:
            # Take first paragraph or first 200 characters
            sentences = self.text_processor.extract_sentences(content)
            if sentences:
                description = sentences[0]
                if len(description) > 200:
                    description = description[:200] + "..."
        
        return description
    
    def _extract_structured_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from item."""
        structured_data = {}
        
        # Extract tables
        tables = item.get('tables', [])
        if tables:
            structured_data['tables'] = tables
        
        # Extract forms
        forms = item.get('forms', [])
        if forms:
            structured_data['forms'] = forms
        
        # Extract links
        links = item.get('links', [])
        if links:
            structured_data['links'] = links
        
        # Extract images
        images = item.get('images', [])
        if images:
            structured_data['images'] = images
        
        # Extract page-specific data
        page_type = item.get('page_type', 'general')
        
        if page_type == 'faq':
            # Extract FAQ data
            faq_data = item.get('questions', [])
            if faq_data:
                structured_data['faq'] = faq_data
        
        elif page_type == 'product':
            # Extract product data
            product_data = {
                'product_name': item.get('product_name', ''),
                'specifications': item.get('specifications', {}),
                'download_links': item.get('download_links', []),
                'parameters': item.get('parameters', []),
            }
            structured_data['product'] = product_data
        
        elif page_type == 'satellite':
            # Extract satellite data
            satellite_data = {
                'satellite_name': item.get('satellite_name', ''),
                'launch_date': item.get('launch_date', ''),
                'instruments': item.get('instruments', []),
                'applications': item.get('applications', []),
                'technical_specs': item.get('technical_specs', {}),
            }
            structured_data['satellite'] = satellite_data
        
        return structured_data
    
    async def _process_text_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process text content through NLP pipeline."""
        main_content = content.get('main_content', '')
        
        if not main_content:
            return content
        
        # Process with text processor
        processed_content = {
            'keywords': self.text_processor.extract_keywords(main_content),
            'named_entities': self.text_processor.extract_named_entities(main_content),
            'technical_terms': self.text_processor.extract_technical_terms(main_content),
            'satellite_info': self.text_processor.extract_satellite_info(main_content),
            'readability': self.text_processor.calculate_readability(main_content),
            'summary': self.text_processor.summarize_text(main_content),
            'qa_pairs': self.text_processor.extract_questions_and_answers(main_content),
            'language': self.text_processor.detect_language(main_content),
            'sentences': self.text_processor.extract_sentences(main_content),
            'chunks': self.text_processor.create_text_chunks(main_content),
        }
        
        # Merge with original content
        return {**content, **processed_content}
    
    def _calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate content quality score."""
        score = 0.0
        
        # Content length factor
        content_length = len(item.get('main_content', ''))
        if content_length > 100:
            length_score = min(content_length / 1000, 1.0)
            score += length_score * 0.2
        
        # Title quality
        title = item.get('title', '')
        if title and len(title) > 10:
            score += 0.1
        
        # Description quality
        description = item.get('description', '')
        if description and len(description) > 20:
            score += 0.1
        
        # Technical content
        technical_terms = item.get('technical_terms', {})
        if technical_terms:
            technical_score = min(len(technical_terms) * 0.1, 0.3)
            score += technical_score
        
        # Structured data
        structured_data = item.get('structured_data', {})
        if structured_data:
            structure_score = min(len(structured_data) * 0.05, 0.2)
            score += structure_score
        
        # Readability
        readability = item.get('readability', {})
        if readability:
            flesch_score = readability.get('flesch_reading_ease', 50)
            readability_score = min(flesch_score / 100, 1.0) * 0.1
            score += readability_score
        
        return min(score, 1.0)
    
    async def batch_process(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Process items in batches."""
        processed_items = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch
            batch_results = await self.process_scraped_data(batch)
            processed_items.extend(batch_results)
            
            # Add delay between batches
            await asyncio.sleep(0.1)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
        
        return processed_items
    
    async def deduplicate_content(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate content based on content hash."""
        seen_hashes = set()
        deduplicated = []
        
        for item in items:
            content_hash = item.get('content_hash')
            if content_hash and content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(item)
            else:
                logger.debug(f"Filtered duplicate content: {item.get('title', 'Unknown')}")
        
        logger.info(f"Deduplicated {len(items)} -> {len(deduplicated)} items")
        return deduplicated
    
    async def filter_by_quality(
        self,
        items: List[Dict[str, Any]],
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Filter items by quality score."""
        filtered = [
            item for item in items
            if item.get('quality_score', 0) >= min_score
        ]
        
        logger.info(f"Quality filtered {len(items)} -> {len(filtered)} items")
        return filtered
    
    async def enrich_with_metadata(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich items with additional metadata."""
        enriched = []
        
        for item in items:
            # Add processing metadata
            item['processing_metadata'] = {
                'processed_at': datetime.utcnow().isoformat(),
                'processor_version': '1.0.0',
                'quality_score': item.get('quality_score', 0),
                'content_length': len(item.get('main_content', '')),
                'keyword_count': len(item.get('keywords', [])),
                'entity_count': sum(len(entities) for entities in item.get('named_entities', {}).values()),
                'technical_term_count': sum(len(terms) for terms in item.get('technical_terms', {}).values()),
            }
            
            # Add content classification
            item['content_classification'] = self._classify_content(item)
            
            # Add SEO metadata
            item['seo_metadata'] = self._extract_seo_metadata(item)
            
            enriched.append(item)
        
        return enriched
    
    def _classify_content(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Classify content type and topic."""
        classification = {
            'primary_topic': 'general',
            'secondary_topics': [],
            'content_type': 'informational',
            'audience': 'general',
            'complexity': 'medium',
        }
        
        # Classify based on technical terms
        technical_terms = item.get('technical_terms', {})
        
        if 'satellites' in technical_terms:
            classification['primary_topic'] = 'satellite'
        elif 'instruments' in technical_terms:
            classification['primary_topic'] = 'instrumentation'
        elif 'applications' in technical_terms:
            classification['primary_topic'] = 'applications'
        elif 'parameters' in technical_terms:
            classification['primary_topic'] = 'data_products'
        
        # Classify content type
        page_type = item.get('page_type', 'general')
        if page_type == 'faq':
            classification['content_type'] = 'faq'
        elif page_type == 'documentation':
            classification['content_type'] = 'documentation'
        elif page_type == 'product':
            classification['content_type'] = 'product_description'
        
        # Classify audience
        readability = item.get('readability', {})
        flesch_score = readability.get('flesch_reading_ease', 50)
        
        if flesch_score > 70:
            classification['audience'] = 'general'
        elif flesch_score > 50:
            classification['audience'] = 'intermediate'
        else:
            classification['audience'] = 'technical'
        
        return classification
    
    def _extract_seo_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SEO-relevant metadata."""
        seo_metadata = {
            'title_length': len(item.get('title', '')),
            'description_length': len(item.get('description', '')),
            'keyword_density': 0,
            'readability_score': 0,
            'content_freshness': 'unknown',
            'internal_links': 0,
            'external_links': 0,
        }
        
        # Calculate keyword density
        keywords = item.get('keywords', [])
        content = item.get('main_content', '')
        if keywords and content:
            word_count = len(content.split())
            keyword_count = sum(content.lower().count(keyword.lower()) for keyword in keywords[:5])
            seo_metadata['keyword_density'] = keyword_count / word_count if word_count > 0 else 0
        
        # Get readability score
        readability = item.get('readability', {})
        seo_metadata['readability_score'] = readability.get('flesch_reading_ease', 0)
        
        # Count links
        links = item.get('links', [])
        domain = item.get('domain', '')
        
        internal_links = sum(1 for link in links if domain in link)
        external_links = len(links) - internal_links
        
        seo_metadata['internal_links'] = internal_links
        seo_metadata['external_links'] = external_links
        
        return seo_metadata
