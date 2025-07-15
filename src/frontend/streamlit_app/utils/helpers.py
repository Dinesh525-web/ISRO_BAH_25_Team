"""
Shared helper utilities.
"""
from datetime import datetime
import traceback
import streamlit as st


def format_datetime(dt: datetime, fmt: str = "%d %b %Y %H:%M") -> str:
    return dt.strftime(fmt)


def handle_error(exc: Exception) -> str:
    tb = traceback.format_exc()
    if st.session_state.user_settings.get("show_debug", False):
        st.error(tb)
    return str(exc)
