"""
Base scraper class for web scraping operations.
"""
import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime

import aiohttp
import asyncio
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from app.core.config import settings
from app.core.exceptions import ScrapingException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseScraper(ABC):
    """Base class for web scrapers."""
    
    def __init__(self, base_url: str, delay: float = 1.0, max_retries: int = 3):
        self.base_url = base_url
        self.delay = delay
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.user_agent = UserAgent()
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.scraped_data: List[Dict[str, Any]] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start aiohttp session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=settings.MOSDAC_API_TIMEOUT)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': self.user_agent.random,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )
    
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_url(self, url: str, retries: int = 0) -> Optional[str]:
        """
        Fetch URL content with retry logic.
        
        Args:
            url: URL to fetch
            retries: Current retry count
            
        Returns:
            HTML content or None if failed
        """
        try:
            if not self.session:
                await self.start_session()
            
            # Add delay between requests
            await asyncio.sleep(self.delay)
            
            logger.info(f"Fetching URL: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    self.visited_urls.add(url)
                    return content
                elif response.status == 429:  # Rate limited
                    if retries < self.max_retries:
                        wait_time = (2 ** retries) * self.delay
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        return await self.fetch_url(url, retries + 1)
                else:
                    logger.error(f"HTTP {response.status} for URL: {url}")
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching URL: {url}")
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            
        # Add to failed URLs
        self.failed_urls.add(url)
        
        # Retry logic
        if retries < self.max_retries:
            logger.info(f"Retrying URL {url} (attempt {retries + 1}/{self.max_retries})")
            return await self.fetch_url(url, retries + 1)
        
        return None
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content."""
        return BeautifulSoup(html, 'html.parser')
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from BeautifulSoup object."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Filter out non-HTTP URLs
            if full_url.startswith(('http://', 'https://')):
                links.append(full_url)
        
        return links
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and within scope."""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(self.base_url)
            
            # Check if URL is within the same domain
            if parsed.netloc != base_parsed.netloc:
                return False
            
            # Check if URL has been visited
            if url in self.visited_urls:
                return False
            
            # Check for common file extensions to skip
            skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.tar', '.gz']
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            return False
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {
            'url': url,
            'title': '',
            'description': '',
            'keywords': '',
            'author': '',
            'published_date': '',
            'scraped_at': datetime.utcnow().isoformat(),
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = content
            elif name == 'author':
                metadata['author'] = content
            elif name in ['date', 'published', 'publication-date']:
                metadata['published_date'] = content
        
        return metadata
    
    @abstractmethod
    async def scrape_page(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single page.
        
        Args:
            url: URL to scrape
            
        Returns:
            Scraped data
        """
        pass
    
    @abstractmethod
    async def scrape_all(self, max_pages: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape multiple pages.
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of scraped data
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        return {
            'total_visited': len(self.visited_urls),
            'total_failed': len(self.failed_urls),
            'total_scraped': len(self.scraped_data),
            'success_rate': len(self.visited_urls) / (len(self.visited_urls) + len(self.failed_urls)) if (len(self.visited_urls) + len(self.failed_urls)) > 0 else 0,
            'failed_urls': list(self.failed_urls),
        }
