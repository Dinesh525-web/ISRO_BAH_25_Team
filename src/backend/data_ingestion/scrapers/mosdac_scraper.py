"""
MOSDAC-specific web scraper for satellite data and documentation.
"""
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

from bs4 import BeautifulSoup, Tag

from .base_scraper import BaseScraper
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MOSDACscraper(BaseScraper):
    """MOSDAC-specific web scraper."""
    
    def __init__(self):
        super().__init__(settings.MOSDAC_BASE_URL, delay=settings.SCRAPING_DELAY)
        self.content_selectors = {
            'main_content': ['#main-content', '.main-content', 'main', '.content'],
            'navigation': ['nav', '.navigation', '.nav-menu'],
            'breadcrumbs': ['.breadcrumb', '.breadcrumbs'],
            'sidebar': ['.sidebar', '.side-menu', 'aside'],
            'footer': ['footer', '.footer'],
        }
    
    async def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single MOSDAC page."""
        try:
            html = await self.fetch_url(url)
            if not html:
                return {}
            
            soup = self.parse_html(html)
            
            # Extract basic information
            page_data = {
                'url': url,
                'content_hash': self.generate_content_hash(html),
                'scraped_at': datetime.utcnow().isoformat(),
                'page_type': self.detect_page_type(soup, url),
                'title': self.extract_title(soup),
                'description': self.extract_description(soup),
                'main_content': self.extract_main_content(soup),
                'navigation': self.extract_navigation(soup),
                'breadcrumbs': self.extract_breadcrumbs(soup),
                'metadata': self.extract_metadata(soup, url),
                'links': self.extract_links(soup, url),
                'images': self.extract_images(soup, url),
                'tables': self.extract_tables(soup),
                'forms': self.extract_forms(soup),
            }
            
            # Extract MOSDAC-specific content
            if page_data['page_type'] == 'product':
                page_data.update(self.extract_product_info(soup))
            elif page_data['page_type'] == 'satellite':
                page_data.update(self.extract_satellite_info(soup))
            elif page_data['page_type'] == 'faq':
                page_data.update(self.extract_faq_content(soup))
            elif page_data['page_type'] == 'documentation':
                page_data.update(self.extract_documentation_content(soup))
            
            return page_data
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return {}
    
    def detect_page_type(self, soup: BeautifulSoup, url: str) -> str:
        """Detect the type of MOSDAC page."""
        url_lower = url.lower()
        
        # Check URL patterns
        if 'product' in url_lower:
            return 'product'
        elif 'satellite' in url_lower:
            return 'satellite'
        elif 'faq' in url_lower:
            return 'faq'
        elif 'documentation' in url_lower or 'manual' in url_lower:
            return 'documentation'
        elif 'news' in url_lower or 'announcement' in url_lower:
            return 'news'
        elif 'service' in url_lower:
            return 'service'
        elif 'about' in url_lower:
            return 'about'
        elif 'contact' in url_lower:
            return 'contact'
        
        # Check content patterns
        title = self.extract_title(soup).lower()
        if 'product' in title:
            return 'product'
        elif 'satellite' in title:
            return 'satellite'
        elif 'faq' in title or 'frequently asked' in title:
            return 'faq'
        
        return 'general'
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try h1 tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return ''
    
    def extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '').strip()
        
        # Try first paragraph
        first_p = soup.find('p')
        if first_p:
            return first_p.get_text().strip()[:200]
        
        return ''
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page."""
        # Try different content selectors
        for selector in self.content_selectors['main_content']:
            content_element = soup.select_one(selector)
            if content_element:
                return self.extract_text(content_element)
        
        # Fallback to body
        body = soup.find('body')
        if body:
            # Remove navigation, footer, and sidebar
            for selector_group in ['navigation', 'footer', 'sidebar']:
                for selector in self.content_selectors[selector_group]:
                    for element in body.select(selector):
                        element.decompose()
            
            return self.extract_text(body)
        
        return ''
    
    def extract_navigation(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract navigation menu."""
        navigation = []
        
        for selector in self.content_selectors['navigation']:
            nav_element = soup.select_one(selector)
            if nav_element:
                for link in nav_element.find_all('a', href=True):
                    navigation.append({
                        'text': link.get_text().strip(),
                        'url': link['href'],
                        'title': link.get('title', ''),
                    })
                break
        
        return navigation
    
    def extract_breadcrumbs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract breadcrumb navigation."""
        breadcrumbs = []
        
        for selector in self.content_selectors['breadcrumbs']:
            breadcrumb_element = soup.select_one(selector)
            if breadcrumb_element:
                for link in breadcrumb_element.find_all('a', href=True):
                    breadcrumbs.append({
                        'text': link.get_text().strip(),
                        'url': link['href'],
                    })
                break
        
        return breadcrumbs
    
    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract image information."""
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                images.append({
                    'src': urljoin(base_url, src),
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width'),
                    'height': img.get('height'),
                })
        
        return images
    
    def extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract table data."""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = {
                'headers': [],
                'rows': [],
                'caption': '',
            }
            
            # Extract caption
            caption = table.find('caption')
            if caption:
                table_data['caption'] = caption.get_text().strip()
            
            # Extract headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    for th in header_row.find_all(['th', 'td']):
                        table_data['headers'].append(th.get_text().strip())
            
            # Extract rows
            tbody = table.find('tbody') or table
            for row in tbody.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text().strip())
                if row_data:
                    table_data['rows'].append(row_data)
            
            tables.append(table_data)
        
        return tables
    
    def extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract form information."""
        forms = []
        
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get'),
                'fields': [],
            }
            
            # Extract form fields
            for field in form.find_all(['input', 'select', 'textarea']):
                field_data = {
                    'name': field.get('name', ''),
                    'type': field.get('type', field.name),
                    'value': field.get('value', ''),
                    'placeholder': field.get('placeholder', ''),
                    'required': field.has_attr('required'),
                }
                
                # Extract options for select fields
                if field.name == 'select':
                    field_data['options'] = []
                    for option in field.find_all('option'):
                        field_data['options'].append({
                            'value': option.get('value', ''),
                            'text': option.get_text().strip(),
                        })
                
                form_data['fields'].append(field_data)
            
            forms.append(form_data)
        
        return forms
    
    def extract_product_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract product-specific information."""
        product_info = {
            'product_name': '',
            'product_description': '',
            'specifications': {},
            'download_links': [],
            'parameters': [],
            'spatial_resolution': '',
            'temporal_resolution': '',
            'spectral_bands': [],
        }
        
        # Extract product name
        product_name_selectors = [
            'h1.product-title',
            '.product-name',
            'h1',
            '.title'
        ]
        
        for selector in product_name_selectors:
            element = soup.select_one(selector)
            if element:
                product_info['product_name'] = element.get_text().strip()
                break
        
        # Extract specifications table
        for table in soup.find_all('table'):
            if 'specification' in table.get('class', []) or 'spec' in str(table).lower():
                specs = {}
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].get_text().strip()
                        value = cells[1].get_text().strip()
                        specs[key] = value
                product_info['specifications'] = specs
                break
        
        # Extract download links
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text().strip().lower()
            
            if any(keyword in link_text for keyword in ['download', 'get data', 'access']):
                product_info['download_links'].append({
                    'url': href,
                    'text': link.get_text().strip(),
                    'title': link.get('title', ''),
                })
        
        return product_info
    
    def extract_satellite_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract satellite-specific information."""
        satellite_info = {
            'satellite_name': '',
            'launch_date': '',
            'mission_life': '',
            'orbit_type': '',
            'instruments': [],
            'applications': [],
            'technical_specs': {},
        }
        
        # Extract satellite name
        title = self.extract_title(soup)
        satellite_info['satellite_name'] = title
        
        # Extract technical specifications
        text_content = self.extract_main_content(soup)
        
        # Look for launch date
        launch_date_pattern = r'launch(?:ed)?\s*(?:date|on)?\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        launch_match = re.search(launch_date_pattern, text_content, re.IGNORECASE)
        if launch_match:
            satellite_info['launch_date'] = launch_match.group(1)
        
        # Extract instruments
        instruments_section = soup.find(text=re.compile(r'instruments?', re.IGNORECASE))
        if instruments_section:
            parent = instruments_section.find_parent()
            if parent:
                instrument_list = parent.find_next('ul') or parent.find_next('ol')
                if instrument_list:
                    for li in instrument_list.find_all('li'):
                        satellite_info['instruments'].append(li.get_text().strip())
        
        return satellite_info
    
    def extract_faq_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract FAQ content."""
        faq_data = {
            'questions': [],
            'categories': [],
        }
        
        # Look for FAQ sections
        faq_sections = soup.find_all(['div', 'section'], class_=re.compile(r'faq', re.IGNORECASE))
        
        for section in faq_sections:
            # Extract questions and answers
            for item in section.find_all(['div', 'li'], class_=re.compile(r'question|faq-item', re.IGNORECASE)):
                question_element = item.find(['h3', 'h4', 'strong', '.question'])
                answer_element = item.find(['p', 'div', '.answer'])
                
                if question_element and answer_element:
                    faq_data['questions'].append({
                        'question': question_element.get_text().strip(),
                        'answer': answer_element.get_text().strip(),
                    })
        
        # If no structured FAQ found, look for Q&A pattern in text
        if not faq_data['questions']:
            text_content = self.extract_main_content(soup)
            qa_pattern = r'Q\s*:?\s*(.+?)\s*A\s*:?\s*(.+?)(?=Q\s*:?|$)'
            qa_matches = re.findall(qa_pattern, text_content, re.IGNORECASE | re.DOTALL)
            
            for question, answer in qa_matches:
                faq_data['questions'].append({
                    'question': question.strip(),
                    'answer': answer.strip(),
                })
        
        return faq_data
    
    def extract_documentation_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract documentation content."""
        doc_data = {
            'sections': [],
            'code_examples': [],
            'diagrams': [],
            'references': [],
        }
        
        # Extract sections
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            section = {
                'title': heading.get_text().strip(),
                'level': int(heading.name[1]),
                'content': '',
            }
            
            # Get content until next heading
            content_elements = []
            for sibling in heading.next_siblings:
                if isinstance(sibling, Tag) and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                if isinstance(sibling, Tag):
                    content_elements.append(sibling)
            
            if content_elements:
                content_soup = BeautifulSoup('<div></div>', 'html.parser')
                for element in content_elements:
                    content_soup.div.append(element)
                section['content'] = self.extract_text(content_soup)
            
            doc_data['sections'].append(section)
        
        # Extract code examples
        for code_block in soup.find_all(['code', 'pre']):
            doc_data['code_examples'].append({
                'code': code_block.get_text().strip(),
                'language': code_block.get('class', [''])[0] if code_block.get('class') else '',
            })
        
        return doc_data
    
    async def scrape_all(self, max_pages: int = 100) -> List[Dict[str, Any]]:
        """Scrape all accessible MOSDAC pages."""
        try:
            # Start with the home page
            urls_to_visit = [self.base_url]
            scraped_count = 0
            
            while urls_to_visit and scraped_count < max_pages:
                url = urls_to_visit.pop(0)
                
                if not self.is_valid_url(url):
                    continue
                
                logger.info(f"Scraping page {scraped_count + 1}/{max_pages}: {url}")
                
                page_data = await self.scrape_page(url)
                if page_data:
                    self.scraped_data.append(page_data)
                    scraped_count += 1
                    
                    # Add new URLs to visit
                    new_urls = page_data.get('links', [])
                    for new_url in new_urls[:10]:  # Limit to prevent explosion
                        if self.is_valid_url(new_url) and new_url not in urls_to_visit:
                            urls_to_visit.append(new_url)
                
                # Respect rate limiting
                await asyncio.sleep(self.delay)
            
            logger.info(f"Scraping completed. Total pages scraped: {len(self.scraped_data)}")
            return self.scraped_data
            
        except Exception as e:
            logger.error(f"Error in scrape_all: {e}")
            return self.scraped_data
    
    async def scrape_specific_sections(self, sections: List[str]) -> List[Dict[str, Any]]:
        """Scrape specific sections of MOSDAC."""
        section_urls = {
            'products': f"{self.base_url}/products",
            'satellites': f"{self.base_url}/satellites",
            'services': f"{self.base_url}/services",
            'faq': f"{self.base_url}/faq",
            'documentation': f"{self.base_url}/documentation",
            'news': f"{self.base_url}/news",
        }
        
        scraped_data = []
        
        for section in sections:
            if section in section_urls:
                url = section_urls[section]
                logger.info(f"Scraping section: {section} ({url})")
                
                page_data = await self.scrape_page(url)
                if page_data:
                    scraped_data.append(page_data)
                    
                    # Scrape linked pages within this section
                    for link in page_data.get('links', [])[:20]:  # Limit per section
                        if section in link.lower() and self.is_valid_url(link):
                            sub_page_data = await self.scrape_page(link)
                            if sub_page_data:
                                scraped_data.append(sub_page_data)
        
        return scraped_data
