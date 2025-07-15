"""
Analytics & monitoring page.
"""
import asyncio
import streamlit as st
import pandas as pd
from utils.api_client import APIClient
from components.visualization import VisualizationComponent


def render_analytics_page():
    api = APIClient()
    st.markdown("### 📊 System Analytics")

    with st.spinner("Loading metrics…"):
        stats = asyncio.run(api.get("/api/v1/metrics"))
        if not stats:
            st.error("Metrics unavailable.")
            return

    st.subheader("Document Status")
    VisualizationComponent.bar_chart(
        [{"status": k, "count": v} for k, v in stats["document_status"].items()],
        x="status",
        y="count",
        title="Document Processing Status",
    )

    st.subheader("Trending Search Queries")
    df = pd.DataFrame(stats["trending_queries"], columns=["query", "hits"])
    VisualizationComponent.bar_chart(df.to_dict("records"), "query", "hits", "Top Queries (30 d)")
