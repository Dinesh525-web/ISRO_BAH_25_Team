"""
Reusable Plotly/Altair visual components.
"""
from typing import List, Dict, Any
import streamlit as st
import altair as alt
import pandas as pd
import plotly.express as px


class VisualizationComponent:
    """Render interactive charts and maps."""

    @staticmethod
    def bar_chart(data: List[Dict[str, Any]], x: str, y: str, title: str):
        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_bar(color="#2a5298")
            .encode(x=x, y=y, tooltip=list(df.columns))
            .properties(title=title, width="container")
        )
        st.altair_chart(chart, use_container_width=True)

    @staticmethod
    def time_series(data: List[Dict[str, Any]], x: str, y: str, title: str):
        df = pd.DataFrame(data)
        fig = px.line(df, x=x, y=y, title=title)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def geo_map(data: pd.DataFrame, lat: str, lon: str, color: str, size: str, title: str):
        fig = px.scatter_mapbox(
            data,
            lat=lat,
            lon=lon,
            color=color,
            size=size,
            zoom=3,
            height=400,
            title=title,
            mapbox_style="carto-positron",
        )
        st.plotly_chart(fig, use_container_width=True)

