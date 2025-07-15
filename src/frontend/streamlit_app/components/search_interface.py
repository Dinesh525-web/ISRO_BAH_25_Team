"""
Search interface component for Streamlit app.
"""
import streamlit as st
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from utils.api_client import APIClient
from utils.helpers import format_datetime, handle_error


class SearchInterface:
    """Search interface component."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.search_history = []
        self.max_history_length = 20
    
    async def search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search request.
        
        Args:
            search_params: Search parameters
            
        Returns:
            Search results
        """
        try:
            # Add to search history
            self.search_history.append({
                'query': search_params.get('query', ''),
                'timestamp': datetime.now(),
                'params': search_params
            })
            
            # Limit history size
            if len(self.search_history) > self.max_history_length:
                self.search_history = self.search_history[-self.max_history_length:]
            
            # Execute search
            response = await self.api_client.post('/search/', search_params)
            
            return response
            
        except Exception as e:
            st.error(f"Search error: {handle_error(e)}")
            return {
                'results': [],
                'total': 0,
                'error': str(e)
            }
    
    def render_search_form(self) -> Dict[str, Any]:
        """
        Render search form and return parameters.
        
        Returns:
            Search parameters
        """
        with st.form(key='advanced_search_form'):
            st.markdown("### 🔍 Advanced Search")
            
            # Main search input
            col1, col2 = st.columns([4, 1])
            
            with col1:
                query = st.text_input(
                    "Search Query",
                    placeholder="Enter your search terms...",
                    key="adv_search_query"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                search_btn = st.form_submit_button("🔍 Search", use_container_width=True)
            
            # Search options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_type = st.selectbox(
                    "Search Type",
                    ["Hybrid", "Semantic", "Keyword"],
                    help="Hybrid combines semantic and keyword search"
                )
            
            with col2:
                limit = st.slider(
                    "Results Limit",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Sort By",
                    ["Relevance", "Date", "Quality", "Title"],
                    help="Sort search results by different criteria"
                )
            
            # Filters
            st.markdown("#### Filters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                categories = st.multiselect(
                    "Categories",
                    ["Satellites", "Weather", "Ocean", "Agriculture", "Disaster", "General"],
                    help="Filter by content category"
                )
                
                content_types = st.multiselect(
                    "Content Types",
                    ["HTML", "PDF", "JSON", "Text"],
                    help="Filter by content type"
                )
            
            with col2:
                date_range = st.date_input(
                    "Date Range",
                    value=[],
                    help="Filter by publication date"
                )
                
                quality_min = st.slider(
                    "Minimum Quality",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Filter by content quality score"
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    include_metadata = st.checkbox(
                        "Include Metadata",
                        value=True,
                        help="Include document metadata in results"
                    )
                    
                    fuzzy_search = st.checkbox(
                        "Fuzzy Search",
                        value=False,
                        help="Enable fuzzy matching for search terms"
                    )
                
                with col2:
                    boost_technical = st.checkbox(
                        "Boost Technical Content",
                        value=True,
                        help="Prioritize technical/scientific content"
                    )
                    
                    group_similar = st.checkbox(
                        "Group Similar Results",
                        value=False,
                        help="Group similar documents together"
                    )
            
            # Prepare search parameters
            search_params = {
                'query': query,
                'search_type': search_type.lower(),
                'limit': limit,
                'offset': 0,
                'include_metadata': include_metadata,
                'filters': {}
            }
            
            # Add filters
            if categories:
                search_params['filters']['categories'] = categories
            
            if content_types:
                search_params['filters']['content_types'] = [ct.lower() for ct in content_types]
            
            if date_range and len(date_range) == 2:
                search_params['filters']['date_from'] = date_range[0].isoformat()
                search_params['filters']['date_to'] = date_range[1].isoformat()
            
            if quality_min > 0:
                search_params['filters']['quality_score_min'] = quality_min
            
            # Add advanced options
            search_params['options'] = {
                'fuzzy_search': fuzzy_search,
                'boost_technical': boost_technical,
                'group_similar': group_similar,
                'sort_by': sort_by.lower()
            }
            
            return search_params if search_btn and query else None
    
    def render_search_results(self, results: Dict[str, Any]):
        """
        Render search results.
        
        Args:
            results: Search results data
        """
        if not results or not results.get('results'):
            st.info("No results found. Try different search terms or filters.")
            return
        
        # Results header
        st.markdown(f"### 📋 Search Results ({results.get('total', 0)} found)")
        
        if results.get('execution_time'):
            st.caption(f"Search completed in {results['execution_time']:.2f} seconds")
        
        # Results list
        for i, result in enumerate(results['results'], 1):
            with st.expander(
                f"{i}. {result.get('title', 'Untitled Document')}",
                expanded=i <= 3
            ):
                # Result header
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Relevance**: {result.get('relevance_score', 0):.2f}")
                
                with col2:
                    st.markdown(f"**Category**: {result.get('category', 'Unknown')}")
                
                with col3:
                    st.markdown(f"**Type**: {result.get('document_type', 'Unknown')}")
                
                # Content preview
                content = result.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                
                st.markdown(f"**Content Preview**:")
                st.markdown(content)
                
                # Metadata
                if result.get('metadata'):
                    with st.expander("Metadata"):
                        st.json(result['metadata'])
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result.get('source_url'):
                        st.markdown(f"[View Source]({result['source_url']})")
                
                with col2:
                    if st.button(f"Find Similar", key=f"similar_{i}"):
                        self.find_similar_documents(result.get('id'))
                
                with col3:
                    if st.button(f"Add to Chat", key=f"chat_{i}"):
                        self.add_to_chat(result)
    
    def render_search_suggestions(self):
        """Render search suggestions."""
        st.markdown("### 💡 Search Suggestions")
        
        suggestions = [
            "INSAT-3D temperature data",
            "Cyclone tracking information",
            "Ocean wind measurements",
            "SCATSAT-1 data products",
            "Monsoon rainfall analysis",
            "Satellite image processing",
            "Weather forecast models",
            "Chlorophyll concentration data"
        ]
        
        # Display suggestions in a grid
        cols = st.columns(4)
        
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 4]
            
            with col:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.adv_search_query = suggestion
                    st.rerun()
    
    def render_search_history(self):
        """Render search history."""
        if not self.search_history:
            st.info("No search history available.")
            return
        
        st.markdown("### 🕒 Recent Searches")
        
        for i, search in enumerate(reversed(self.search_history[-10:])):
            with st.expander(f"{search['query']} - {format_datetime(search['timestamp'])}"):
                st.markdown(f"**Query**: {search['query']}")
                st.markdown(f"**Type**: {search['params'].get('search_type', 'unknown')}")
                st.markdown(f"**Filters**: {search['params'].get('filters', {})}")
                
                if st.button(f"Repeat Search", key=f"repeat_{i}"):
                    # Re-execute search
                    results = asyncio.run(self.search(search['params']))
                    self.render_search_results(results)
    
    def render_filter_statistics(self):
        """Render filter statistics."""
        st.markdown("### 📊 Filter Statistics")
        
        try:
            # Get filter statistics from API
            stats = asyncio.run(self.api_client.get('/search/filters'))
            
            if stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'categories' in stats:
                        st.markdown("**Categories**:")
                        for category in stats['categories']:
                            st.markdown(f"- {category}")
                    
                    if 'content_types' in stats:
                        st.markdown("**Content Types**:")
                        for content_type in stats['content_types']:
                            st.markdown(f"- {content_type}")
                
                with col2:
                    if 'date_ranges' in stats:
                        date_info = stats['date_ranges']
                        st.markdown("**Date Range**:")
                        st.markdown(f"- From: {date_info.get('min_date', 'Unknown')}")
                        st.markdown(f"- To: {date_info.get('max_date', 'Unknown')}")
                    
                    if 'languages' in stats:
                        st.markdown("**Languages**:")
                        for language in stats['languages']:
                            st.markdown(f"- {language}")
        
        except Exception as e:
            st.error(f"Could not load filter statistics: {handle_error(e)}")
    
    def find_similar_documents(self, document_id: str):
        """Find similar documents."""
        try:
            similar_results = asyncio.run(
                self.api_client.get(f'/search/similar/{document_id}')
            )
            
            if similar_results:
                st.markdown("### 📄 Similar Documents")
                self.render_search_results({'results': similar_results})
            else:
                st.info("No similar documents found.")
                
        except Exception as e:
            st.error(f"Error finding similar documents: {handle_error(e)}")
    
    def add_to_chat(self, result: Dict[str, Any]):
        """Add search result to chat."""
        title = result.get('title', 'Document')
        content = result.get('content', '')[:200] + "..."
        
        chat_message = f"Tell me more about: {title}\n\nContent preview: {content}"
        
        # Add to chat input
        st.session_state.chat_input = chat_message
        st.session_state.current_page = 'Chat'
        st.rerun()
    
    def export_search_results(self, results: Dict[str, Any]) -> str:
        """Export search results to text format."""
        if not results or not results.get('results'):
            return "No search results to export."
        
        export_content = f"""# Search Results Export

**Exported on**: {format_datetime(datetime.now())}
**Query**: {results.get('query', 'Unknown')}
**Total Results**: {results.get('total', 0)}
**Execution Time**: {results.get('execution_time', 0):.2f}s

---

"""
        
        for i, result in enumerate(results['results'], 1):
            title = result.get('title', 'Untitled')
            content = result.get('content', '')
            relevance = result.get('relevance_score', 0)
            category = result.get('category', 'Unknown')
            source_url = result.get('source_url', '')
            
            export_content += f"""## {i}. {title}

**Relevance**: {relevance:.2f}
**Category**: {category}
**Source**: {source_url}

**Content**:
{content}

---

"""
        
        return export_content
