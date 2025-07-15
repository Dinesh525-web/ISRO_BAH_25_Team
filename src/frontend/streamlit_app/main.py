"""
Main Streamlit application for MOSDAC AI Knowledge Navigator.
"""
import streamlit as st
import asyncio
from datetime import datetime
import json
from typing import Dict, Any, List, Optional

from components.chat_interface import ChatInterface
from components.search_interface import SearchInterface
from components.visualization import VisualizationComponent
from utils.api_client import APIClient
from utils.helpers import format_datetime, handle_error
from pages.home import render_home_page
from pages.analytics import render_analytics_page
from pages.settings import render_settings_page

# Page configuration
st.set_page_config(
    page_title="MOSDAC AI Knowledge Navigator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.mosdac.gov.in/help',
        'Report a bug': 'https://github.com/gravitasops/mosdac-ai/issues',
        'About': """
        # MOSDAC AI Knowledge Navigator
        
        An intelligent AI-powered conversational assistant for the MOSDAC portal.
        Built with RAG and knowledge graphs for accurate satellite data queries.
        
        **Version**: 1.0.0  
        **Developer**: Team GravitasOps
        """
    }
)

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {
        'theme': 'light',
        'language': 'en',
        'show_debug': False,
        'auto_scroll': True,
        'search_limit': 10,
    }

def main():
    """Main application entry point."""
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
        }
        
        .user-message {
            background: #e3f2fd;
            border-left-color: #1976d2;
        }
        
        .assistant-message {
            background: #f3e5f5;
            border-left-color: #7b1fa2;
        }
        
        .sidebar-info {
            background: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background-color: #4caf50;
        }
        
        .status-offline {
            background-color: #f44336;
        }
        
        .alert-info {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background-color: #2a5298;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background-color: #1e3c72;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ MOSDAC AI Knowledge Navigator</h1>
        <p>Intelligent conversational assistant for satellite data and meteorological information</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Main content
    page = st.session_state.get('current_page', 'Home')
    
    if page == 'Home':
        render_home_page()
    elif page == 'Chat':
        render_chat_page()
    elif page == 'Search':
        render_search_page()
    elif page == 'Analytics':
        render_analytics_page()
    elif page == 'Settings':
        render_settings_page()
    
    # Footer
    render_footer()

def render_sidebar():
    """Render sidebar with navigation and system info."""
    
    with st.sidebar:
        st.markdown("## Navigation")
        
        # Navigation buttons
        pages = ['Home', 'Chat', 'Search', 'Analytics', 'Settings']
        
        for page in pages:
            icon = {
                'Home': '🏠',
                'Chat': '💬',
                'Search': '🔍',
                'Analytics': '📊',
                'Settings': '⚙️'
            }.get(page, '📄')
            
            if st.button(f"{icon} {page}", key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("## System Status")
        
        # Check API health
        try:
            health_status = asyncio.run(st.session_state.api_client.health_check())
            if health_status.get('status') == 'healthy':
                st.markdown(
                    '<div class="status-indicator status-online"></div>API: Online',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="status-indicator status-offline"></div>API: Offline',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.markdown(
                '<div class="status-indicator status-offline"></div>API: Error',
                unsafe_allow_html=True
            )
        
        # System info
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("**System Information**")
        st.markdown(f"**Version**: 1.0.0")
        st.markdown(f"**Updated**: {format_datetime(datetime.now())}")
        st.markdown(f"**Sessions**: {len(st.session_state.chat_history)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("## Quick Actions")
        
        if st.button("🔄 Refresh", key="refresh_btn"):
            st.rerun()
        
        if st.button("🗑️ Clear History", key="clear_history_btn"):
            st.session_state.chat_history = []
            st.session_state.current_session_id = None
            st.success("History cleared!")
        
        if st.button("📥 Export Chat", key="export_chat_btn"):
            export_chat_history()
        
        # Debug information
        if st.session_state.user_settings.get('show_debug', False):
            st.markdown("## Debug Info")
            st.json({
                "session_id": st.session_state.current_session_id,
                "chat_length": len(st.session_state.chat_history),
                "settings": st.session_state.user_settings,
            })

def render_chat_page():
    """Render chat interface page."""
    
    st.markdown("## 💬 Chat with MOSDAC AI")
    
    # Chat interface
    chat_interface = ChatInterface(st.session_state.api_client)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            message_class = "user-message" if message['type'] == 'user' else "assistant-message"
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <strong>{'You' if message['type'] == 'user' else 'MOSDAC AI'}:</strong><br>
                {message['content']}
                <br><small>{format_datetime(message['timestamp'])}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Auto-scroll to bottom
        if st.session_state.user_settings.get('auto_scroll', True):
            st.empty()
    
    # Input form
    with st.form(key='chat_form', clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_area(
                "Ask me anything about MOSDAC, satellites, or meteorological data:",
                placeholder="e.g., What is the latest cyclone information?",
                height=100,
                key="chat_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Send 🚀")
            
            # Advanced options
            with st.expander("Advanced Options"):
                include_sources = st.checkbox("Include sources", value=True)
                response_length = st.selectbox(
                    "Response length",
                    ["Short", "Medium", "Long"],
                    index=1
                )
        
        if submit_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Show thinking indicator
            with st.spinner("MOSDAC AI is thinking..."):
                try:
                    # Send request to API
                    response = asyncio.run(
                        chat_interface.send_message(
                            user_input,
                            st.session_state.current_session_id,
                            include_sources=include_sources,
                            response_length=response_length
                        )
                    )
                    
                    # Update session ID
                    st.session_state.current_session_id = response.get('session_id')
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': response.get('content', 'Sorry, I encountered an error.'),
                        'timestamp': datetime.now(),
                        'metadata': response.get('metadata', {})
                    })
                    
                    # Show response metrics
                    if response.get('confidence_score'):
                        st.info(f"Confidence: {response['confidence_score']:.2f}")
                    
                    if response.get('processing_time'):
                        st.info(f"Response time: {response['processing_time']:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error: {handle_error(e)}")
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': 'Sorry, I encountered an error processing your request.',
                        'timestamp': datetime.now()
                    })
            
            st.rerun()

def render_search_page():
    """Render search interface page."""
    
    st.markdown("## 🔍 Search MOSDAC Knowledge Base")
    
    # Search interface
    search_interface = SearchInterface(st.session_state.api_client)
    
    # Search form
    with st.form(key='search_form'):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="e.g., INSAT-3D temperature data",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.form_submit_button("Search")
        
        # Search options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_type = st.selectbox(
                "Search Type",
                ["Hybrid", "Semantic", "Keyword"],
                key="search_type"
            )
        
        with col2:
            result_limit = st.slider(
                "Max Results",
                min_value=5,
                max_value=50,
                value=10,
                key="result_limit"
            )
        
        with col3:
            category_filter = st.selectbox(
                "Category",
                ["All", "Satellites", "Weather", "Ocean", "Agriculture"],
                key="category_filter"
            )
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                date_range = st.date_input(
                    "Date Range",
                    value=[],
                    key="date_range"
                )
            
            with col2:
                quality_threshold = st.slider(
                    "Quality Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    key="quality_threshold"
                )
        
        if search_button and search_query:
            with st.spinner("Searching..."):
                try:
                    # Prepare search parameters
                    search_params = {
                        'query': search_query,
                        'search_type': search_type.lower(),
                        'limit': result_limit,
                        'filters': {}
                    }
                    
                    if category_filter != "All":
                        search_params['filters']['category'] = category_filter
                    
                    if quality_threshold > 0:
                        search_params['filters']['quality_score_min'] = quality_threshold
                    
                    # Execute search
                    results = asyncio.run(
                        search_interface.search(search_params)
                    )
                    
                    # Display results
                    if results.get('results'):
                        st.success(f"Found {len(results['results'])} results in {results.get('execution_time', 0):.2f}s")
                        
                        for i, result in enumerate(results['results'], 1):
                            with st.expander(f"{i}. {result.get('title', 'Untitled')}", expanded=i<=3):
                                st.markdown(f"**Relevance**: {result.get('relevance_score', 0):.2f}")
                                st.markdown(f"**Category**: {result.get('category', 'Unknown')}")
                                st.markdown(f"**Content**: {result.get('content', '')[:300]}...")
                                
                                if result.get('source_url'):
                                    st.markdown(f"**Source**: [Link]({result['source_url']})")
                                
                                if result.get('tags'):
                                    st.markdown(f"**Tags**: {', '.join(result['tags'])}")
                    else:
                        st.warning("No results found. Try different search terms.")
                
                except Exception as e:
                    st.error(f"Search error: {handle_error(e)}")

def render_footer():
    """Render application footer."""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**MOSDAC AI Navigator**")
        st.markdown("Built for ISRO Bharatiya Antariksh Hackathon 2025")
    
    with col2:
        st.markdown("**Quick Links**")
        st.markdown("- [MOSDAC Portal](https://www.mosdac.gov.in)")
        st.markdown("- [API Documentation](http://localhost:8000/docs)")
        st.markdown("- [GitHub Repository](https://github.com/gravitasops/mosdac-ai)")
    
    with col3:
        st.markdown("**Support**")
        st.markdown("- [Help Center](https://www.mosdac.gov.in/help)")
        st.markdown("- [Contact Us](mailto:support@gravitasops.com)")
        st.markdown("- [Report Issues](https://github.com/gravitasops/mosdac-ai/issues)")

def export_chat_history():
    """Export chat history to JSON."""
    
    if not st.session_state.chat_history:
        st.warning("No chat history to export.")
        return
    
    # Prepare export data
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'session_id': st.session_state.current_session_id,
        'message_count': len(st.session_state.chat_history),
        'messages': [
            {
                'type': msg['type'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat(),
                'metadata': msg.get('metadata', {})
            }
            for msg in st.session_state.chat_history
        ]
    }
    
    # Create download
    json_data = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="Download Chat History",
        data=json_data,
        file_name=f"mosdac_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
