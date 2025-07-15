"""
Home (dashboard) page.
"""
import streamlit as st
from datetime import datetime
from utils.helpers import format_datetime


def render_home_page():
    st.markdown("### 🏠 Welcome to **MOSDAC AI Navigator**")

    st.write(
        "Use the sidebar to start chatting with the assistant, search the knowledge base, "
        "or view analytics. This dashboard shows system status and quick links."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Uptime", "99.97 %")
    col2.metric("Docs Indexed", "18 271")
    col3.metric("KG Nodes", "12 540")

    st.info(
        f"Last refresh: {format_datetime(datetime.utcnow())} (UTC)",
        icon="ℹ️",
    )
