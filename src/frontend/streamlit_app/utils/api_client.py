"""
API client for communicating with the backend.
"""
import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime

import streamlit as st


class APIClient:
    """API client for backend communication."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'User-Agent': 'MOSDAC-AI-Navigator/1.0'
                }
            )
        return self.session
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Response data
        """
        try:
            session = await self.get_session()
            url = f"{self.base_url}{endpoint}"
            
            # Prepare request arguments
            kwargs = {}
            
            if data:
                kwargs['json'] = data
            
            if params:
                kwargs['params'] = params
            
            if headers:
                kwargs['headers'] = headers
            
            # Make request
            async with session.request(method, url, **kwargs) as response:
                response_text = await response.text()
                
                # Handle different response types
                if response.content_type == 'application/json':
                    response_data = json.loads(response_text)
                else:
                    response_data = {'content': response_text}
                
                # Handle error responses
                if response.status >= 400:
                    error_msg = response_data.get('detail', f'HTTP {response.status}')
                    raise Exception(f"API Error: {error_msg}")
                
                return response_data
                
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request('GET', endpoint, params=params, headers=headers)
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request('POST', endpoint, data=data, headers=headers)
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request('PUT', endpoint, data=data, headers=headers)
    
    async def delete(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request('DELETE', endpoint, headers=headers)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            return await self.get('/api/v1/health/')
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return await self.get('/api/v1/health/detailed')
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_search_suggestions(self, query: str) -> list:
        """Get search suggestions."""
        try:
            params = {'q': query, 'limit': 5}
            response = await self.get('/api/v1/search/suggestions', params=params)
            return response if isinstance(response, list) else []
        except Exception:
            return []
    
    async def get_trending_queries(self) -> list:
        """Get trending search queries."""
        try:
            response = await self.get('/api/v1/search/trending')
            return response if isinstance(response, list) else []
        except Exception:
            return []
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close_session())
            else:
                loop.run_until_complete(self.close_session())
