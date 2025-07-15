"""
Real-time scraper for dynamic MOSDAC content.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

from .base_scraper import BaseScraper
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeScraper(BaseScraper):
    """Real-time scraper for dynamic MOSDAC content."""
    
    def __init__(self):
        super().__init__(settings.MOSDAC_BASE_URL, delay=settings.SCRAPING_DELAY)
        self.driver: Optional[webdriver.Chrome] = None
        self.wait_timeout = 10
        self.api_endpoints = {
            'current_weather': f"{self.base_url}/api/weather/current",
            'satellite_status': f"{self.base_url}/api/satellites/status",
            'data_updates': f"{self.base_url}/api/data/updates",
            'system_status': f"{self.base_url}/api/system/status",
        }
    
    def setup_driver(self) -> webdriver.Chrome:
        """Setup Selenium WebDriver."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(self.wait_timeout)
            
            return driver
            
        except Exception as e:
            logger.error(f"Error setting up WebDriver: {e}")
            raise
    
    async def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page using Selenium."""
        try:
            if not self.driver:
                self.driver = self.setup_driver()
            
            logger.info(f"Scraping dynamic content from: {url}")
            
            # Navigate to the page
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.wait_timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            await asyncio.sleep(3)
            
            # Get page source after JavaScript execution
            html = self.driver.page_source
            soup = self.parse_html(html)
            
            # Extract data
            page_data = {
                'url': url,
                'content_hash': self.generate_content_hash(html),
                'scraped_at': datetime.utcnow().isoformat(),
                'page_type': 'dynamic',
                'title': self.driver.title,
                'main_content': self.extract_text(soup),
                'dynamic_elements': self.extract_dynamic_elements(),
                'javascript_data': self.extract_javascript_data(),
                'api_data': await self.extract_api_data(),
            }
            
            return page_data
            
        except TimeoutException:
            logger.error(f"Timeout loading page: {url}")
            return {}
        except WebDriverException as e:
            logger.error(f"WebDriver error for {url}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error scraping dynamic page {url}: {e}")
            return {}
    
    def extract_dynamic_elements(self) -> List[Dict[str, Any]]:
        """Extract dynamic elements from the page."""
        dynamic_elements = []
        
        try:
            # Extract elements with dynamic content
            dynamic_selectors = [
                {'selector': '.realtime-data', 'type': 'realtime'},
                {'selector': '.weather-widget', 'type': 'weather'},
                {'selector': '.satellite-status', 'type': 'satellite'},
                {'selector': '.data-counter', 'type': 'counter'},
                {'selector': '.live-feed', 'type': 'feed'},
                {'selector': '.dashboard-widget', 'type': 'widget'},
            ]
            
            for selector_info in dynamic_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector_info['selector'])
                
                for element in elements:
                    dynamic_elements.append({
                        'type': selector_info['type'],
                        'selector': selector_info['selector'],
                        'text': element.text,
                        'attributes': element.get_attribute('outerHTML'),
                        'location': element.location,
                        'size': element.size,
                    })
        
        except Exception as e:
            logger.error(f"Error extracting dynamic elements: {e}")
        
        return dynamic_elements
    
    def extract_javascript_data(self) -> Dict[str, Any]:
        """Extract data from JavaScript variables."""
        js_data = {}
        
        try:
            # Common JavaScript variables to extract
            js_variables = [
                'window.appData',
                'window.configData',
                'window.satelliteData',
                'window.weatherData',
                'window.realtimeData',
            ]
            
            for var_name in js_variables:
                try:
                    result = self.driver.execute_script(f"return {var_name};")
                    if result:
                        js_data[var_name] = result
                except Exception:
                    continue
            
            # Extract from script tags
            script_tags = self.driver.find_elements(By.TAG_NAME, "script")
            for script in script_tags:
                script_content = script.get_attribute('innerHTML')
                if script_content and 'var ' in script_content:
                    # Extract variable definitions
                    var_pattern = r'var\s+(\w+)\s*=\s*({[^}]+}|\[[^\]]+\]|"[^"]*"|\'[^\']*\'|\d+)'
                    import re
                    matches = re.findall(var_pattern, script_content)
                    for var_name, var_value in matches:
                        try:
                            js_data[f"script_{var_name}"] = json.loads(var_value)
                        except:
                            js_data[f"script_{var_name}"] = var_value
        
        except Exception as e:
            logger.error(f"Error extracting JavaScript data: {e}")
        
        return js_data
    
    async def extract_api_data(self) -> Dict[str, Any]:
        """Extract data from API endpoints."""
        api_data = {}
        
        if not self.session:
            await self.start_session()
        
        for endpoint_name, endpoint_url in self.api_endpoints.items():
            try:
                async with self.session.get(endpoint_url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        
                        if 'application/json' in content_type:
                            data = await response.json()
                            api_data[endpoint_name] = data
                        else:
                            text = await response.text()
                            api_data[endpoint_name] = text
                    else:
                        logger.warning(f"API endpoint {endpoint_name} returned {response.status}")
                        
            except Exception as e:
                logger.error(f"Error fetching API data from {endpoint_name}: {e}")
        
        return api_data
    
    async def scrape_weather_data(self) -> Dict[str, Any]:
        """Scrape real-time weather data."""
        weather_data = {
            'current_conditions': {},
            'forecasts': [],
            'alerts': [],
            'satellite_imagery': [],
        }
        
        try:
            # Navigate to weather page
            weather_url = f"{self.base_url}/weather"
            if not self.driver:
                self.driver = self.setup_driver()
            
            self.driver.get(weather_url)
            
            # Wait for weather widgets to load
            WebDriverWait(self.driver, self.wait_timeout).until(
                EC.presence_of_element_located((By.CLASS_NAME, "weather-widget"))
            )
            
            # Extract current conditions
            current_elements = self.driver.find_elements(By.CLASS_NAME, "current-weather")
            for element in current_elements:
                weather_data['current_conditions'] = {
                    'temperature': self.extract_weather_value(element, '.temperature'),
                    'humidity': self.extract_weather_value(element, '.humidity'),
                    'pressure': self.extract_weather_value(element, '.pressure'),
                    'wind_speed': self.extract_weather_value(element, '.wind-speed'),
                    'wind_direction': self.extract_weather_value(element, '.wind-direction'),
                    'visibility': self.extract_weather_value(element, '.visibility'),
                    'conditions': self.extract_weather_value(element, '.conditions'),
                }
            
            # Extract forecasts
            forecast_elements = self.driver.find_elements(By.CLASS_NAME, "forecast-item")
            for element in forecast_elements:
                forecast = {
                    'date': self.extract_weather_value(element, '.date'),
                    'high_temp': self.extract_weather_value(element, '.high-temp'),
                    'low_temp': self.extract_weather_value(element, '.low-temp'),
                    'conditions': self.extract_weather_value(element, '.conditions'),
                    'precipitation': self.extract_weather_value(element, '.precipitation'),
                }
                weather_data['forecasts'].append(forecast)
            
            # Extract alerts
            alert_elements = self.driver.find_elements(By.CLASS_NAME, "weather-alert")
            for element in alert_elements:
                alert = {
                    'type': self.extract_weather_value(element, '.alert-type'),
                    'severity': self.extract_weather_value(element, '.severity'),
                    'message': self.extract_weather_value(element, '.alert-message'),
                    'issued_at': self.extract_weather_value(element, '.issued-time'),
                    'expires_at': self.extract_weather_value(element, '.expires-time'),
                }
                weather_data['alerts'].append(alert)
        
        except Exception as e:
            logger.error(f"Error scraping weather data: {e}")
        
        return weather_data
    
    def extract_weather_value(self, parent_element, selector: str) -> str:
        """Extract weather value from element."""
        try:
            element = parent_element.find_element(By.CSS_SELECTOR, selector)
            return element.text.strip()
        except:
            return ""
    
    async def scrape_satellite_status(self) -> Dict[str, Any]:
        """Scrape satellite status information."""
        satellite_status = {
            'operational_satellites': [],
            'satellite_passes': [],
            'data_availability': {},
            'system_health': {},
        }
        
        try:
            # Navigate to satellite status page
            status_url = f"{self.base_url}/satellites/status"
            if not self.driver:
                self.driver = self.setup_driver()
            
            self.driver.get(status_url)
            
            # Wait for status dashboard to load
            WebDriverWait(self.driver, self.wait_timeout).until(
                EC.presence_of_element_located((By.CLASS_NAME, "satellite-dashboard"))
            )
            
            # Extract operational satellites
            satellite_elements = self.driver.find_elements(By.CLASS_NAME, "satellite-card")
            for element in satellite_elements:
                satellite_info = {
                    'name': self.extract_satellite_value(element, '.satellite-name'),
                    'status': self.extract_satellite_value(element, '.status'),
                    'last_contact': self.extract_satellite_value(element, '.last-contact'),
                    'orbit_position': self.extract_satellite_value(element, '.orbit-position'),
                    'data_quality': self.extract_satellite_value(element, '.data-quality'),
                    'instruments': self.extract_satellite_instruments(element),
                }
                satellite_status['operational_satellites'].append(satellite_info)
            
            # Extract satellite passes
            pass_elements = self.driver.find_elements(By.CLASS_NAME, "satellite-pass")
            for element in pass_elements:
                pass_info = {
                    'satellite': self.extract_satellite_value(element, '.satellite-name'),
                    'start_time': self.extract_satellite_value(element, '.start-time'),
                    'end_time': self.extract_satellite_value(element, '.end-time'),
                    'elevation': self.extract_satellite_value(element, '.elevation'),
                    'azimuth': self.extract_satellite_value(element, '.azimuth'),
                    'visibility': self.extract_satellite_value(element, '.visibility'),
                }
                satellite_status['satellite_passes'].append(pass_info)
        
        except Exception as e:
            logger.error(f"Error scraping satellite status: {e}")
        
        return satellite_status
    
    def extract_satellite_value(self, parent_element, selector: str) -> str:
        """Extract satellite value from element."""
        try:
            element = parent_element.find_element(By.CSS_SELECTOR, selector)
            return element.text.strip()
        except:
            return ""
    
    def extract_satellite_instruments(self, parent_element) -> List[str]:
        """Extract satellite instruments."""
        instruments = []
        try:
            instrument_elements = parent_element.find_elements(By.CLASS_NAME, "instrument")
            for element in instrument_elements:
                instruments.append(element.text.strip())
        except:
            pass
        return instruments
    
    async def scrape_all(self, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Scrape all real-time data sources."""
        try:
            realtime_data = []
            
            # Scrape weather data
            logger.info("Scraping real-time weather data")
            weather_data = await self.scrape_weather_data()
            if weather_data:
                realtime_data.append({
                    'type': 'weather',
                    'data': weather_data,
                    'scraped_at': datetime.utcnow().isoformat(),
                })
            
            # Scrape satellite status
            logger.info("Scraping satellite status data")
            satellite_data = await self.scrape_satellite_status()
            if satellite_data:
                realtime_data.append({
                    'type': 'satellite_status',
                    'data': satellite_data,
                    'scraped_at': datetime.utcnow().isoformat(),
                })
            
            # Scrape dynamic pages
            dynamic_urls = [
                f"{self.base_url}/dashboard",
                f"{self.base_url}/live-data",
                f"{self.base_url}/monitoring",
                f"{self.base_url}/alerts",
            ]
            
            for url in dynamic_urls:
                logger.info(f"Scraping dynamic page: {url}")
                page_data = await self.scrape_page(url)
                if page_data:
                    realtime_data.append(page_data)
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"Error in scrape_all: {e}")
            return []
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    async def monitor_changes(self, urls: List[str], interval: int = 300) -> None:
        """Monitor URLs for changes."""
        previous_hashes = {}
        
        while True:
            try:
                for url in urls:
                    page_data = await self.scrape_page(url)
                    if page_data:
                        current_hash = page_data.get('content_hash')
                        
                        if url in previous_hashes:
                            if current_hash != previous_hashes[url]:
                                logger.info(f"Change detected in {url}")
                                # Store the changed content
                                self.scraped_data.append({
                                    'change_detected': True,
                                    'url': url,
                                    'previous_hash': previous_hashes[url],
                                    'current_hash': current_hash,
                                    'data': page_data,
                                })
                        
                        previous_hashes[url] = current_hash
                
                # Wait before next check
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in change monitoring: {e}")
                await asyncio.sleep(interval)
    
    def __del__(self):
        """Cleanup WebDriver on deletion."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
