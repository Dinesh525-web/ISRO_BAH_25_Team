"""
Chat interface component for Streamlit app.
"""
import streamlit as st
import asyncio
from typing import Dict, Any, Optional
from uuid import uuid4

from utils.api_client import APIClient
from utils.helpers import format_datetime, handle_error


class ChatInterface:
    """Chat interface component."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.max_history_length = 50
    
    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        include_sources: bool = True,
        response_length: str = "Medium"
    ) -> Dict[str, Any]:
        """
        Send message to chat API.
        
        Args:
            message: User message
            session_id: Chat session ID
            include_sources: Whether to include source citations
            response_length: Response length preference
            
        Returns:
            Chat response
        """
        try:
            # Prepare request
            request_data = {
                'message': message,
                'session_id': session_id or str(uuid4()),
                'context': {
                    'include_sources': include_sources,
                    'response_length': response_length.lower(),
                    'user_preferences': st.session_state.user_settings
                },
                'stream': False
            }
            
            # Send request
            response = await self.api_client.post('/chat/', request_data)
            
            return response
            
        except Exception as e:
            st.error(f"Error sending message: {handle_error(e)}")
            return {
                'content': 'Sorry, I encountered an error processing your request.',
                'session_id': session_id or str(uuid4()),
                'error': str(e)
            }
    
    def render_message(self, message: Dict[str, Any], is_user: bool = False):
        """
        Render a chat message.
        
        Args:
            message: Message data
            is_user: Whether this is a user message
        """
        message_class = "user-message" if is_user else "assistant-message"
        sender = "You" if is_user else "MOSDAC AI"
        
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{sender}:</strong><br>
            {message.get('content', '')}
            <br><small>{format_datetime(message.get('timestamp'))}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Show metadata for assistant messages
        if not is_user and message.get('metadata'):
            metadata = message['metadata']
            
            if metadata.get('confidence_score'):
                st.caption(f"Confidence: {metadata['confidence_score']:.2f}")
            
            if metadata.get('processing_time'):
                st.caption(f"Processing time: {metadata['processing_time']:.2f}s")
            
            if metadata.get('retrieved_documents'):
                with st.expander("Sources"):
                    for i, doc in enumerate(metadata['retrieved_documents'], 1):
                        st.markdown(f"**{i}.** {doc.get('title', 'Untitled')}")
                        if doc.get('source_url'):
                            st.markdown(f"[Link]({doc['source_url']})")
    
    def render_typing_indicator(self):
        """Render typing indicator."""
        st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <small>MOSDAC AI is typing...</small>
        </div>
        """, unsafe_allow_html=True)
    
    def get_suggested_questions(self) -> list:
        """Get suggested questions for the user."""
        return [
            "What is the latest weather information?",
            "How can I download INSAT-3D data?",
            "What satellites are currently operational?",
            "Show me cyclone tracking information",
            "How do I access ocean wind data?",
            "What is the difference between INSAT and SCATSAT?",
            "Explain SST data products",
            "How to interpret satellite imagery?",
        ]
    
    def render_suggestions(self):
        """Render suggested questions."""
        st.markdown("### 💡 Suggested Questions")
        
        suggestions = self.get_suggested_questions()
        
        # Display in columns
        cols = st.columns(2)
        
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            
            with col:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    # Add to chat input
                    st.session_state.chat_input = suggestion
                    st.rerun()
    
    def render_quick_actions(self):
        """Render quick action buttons."""
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Latest Data", key="latest_data"):
                st.session_state.chat_input = "What is the latest satellite data available?"
                st.rerun()
        
        with col2:
            if st.button("🌪️ Weather Alerts", key="weather_alerts"):
                st.session_state.chat_input = "Show me current weather alerts and warnings"
                st.rerun()
        
        with col3:
            if st.button("🛰️ Satellite Status", key="satellite_status"):
                st.session_state.chat_input = "What is the current status of ISRO satellites?"
                st.rerun()
    
    def render_chat_stats(self):
        """Render chat statistics."""
        if not st.session_state.chat_history:
            return
        
        st.markdown("### 📈 Chat Statistics")
        
        total_messages = len(st.session_state.chat_history)
        user_messages = sum(1 for msg in st.session_state.chat_history if msg['type'] == 'user')
        assistant_messages = total_messages - user_messages
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", total_messages)
        
        with col2:
            st.metric("Your Messages", user_messages)
        
        with col3:
            st.metric("AI Responses", assistant_messages)
        
        # Recent activity
        if st.session_state.chat_history:
            last_message = st.session_state.chat_history[-1]
            st.caption(f"Last activity: {format_datetime(last_message['timestamp'])}")
    
    def export_conversation(self) -> str:
        """Export conversation to markdown format."""
        if not st.session_state.chat_history:
            return "No conversation history available."
        
        markdown_content = f"""# MOSDAC AI Conversation Export

**Exported on**: {format_datetime(datetime.now())}
**Session ID**: {st.session_state.current_session_id}
**Total Messages**: {len(st.session_state.chat_history)}

---

"""
        
        for message in st.session_state.chat_history:
            sender = "**You**" if message['type'] == 'user' else "**MOSDAC AI**"
            timestamp = format_datetime(message['timestamp'])
            content = message['content']
            
            markdown_content += f"""{sender} ({timestamp}):

{content}

---

"""
        
        return markdown_content
    
    def clear_history(self):
        """Clear chat history."""
        st.session_state.chat_history = []
        st.session_state.current_session_id = None
        st.success("Chat history cleared!")
    
    def get_context_for_message(self, message: str) -> Dict[str, Any]:
        """Get context for a message based on chat history."""
        context = {
            'previous_messages': [],
            'topics_discussed': [],
            'user_preferences': st.session_state.user_settings
        }
        
        # Add recent messages for context
        recent_messages = st.session_state.chat_history[-5:]
        
        for msg in recent_messages:
            context['previous_messages'].append({
                'type': msg['type'],
                'content': msg['content'][:200],  # Truncate for context
                'timestamp': msg['timestamp'].isoformat()
            })
        
        # Extract topics from recent messages
        # This is a simplified implementation
        topics = set()
        for msg in recent_messages:
            content = msg['content'].lower()
            if 'satellite' in content:
                topics.add('satellites')
            if 'weather' in content:
                topics.add('weather')
            if 'ocean' in content:
                topics.add('ocean')
            if 'cyclone' in content:
                topics.add('cyclone')
        
        context['topics_discussed'] = list(topics)
        
        return context
