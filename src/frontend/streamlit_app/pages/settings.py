"""
User-level settings page.
"""
import streamlit as st


def render_settings_page():
    st.markdown("### ⚙️ Settings")

    theme = st.selectbox("Theme", ["light", "dark"], index=0)
    language = st.selectbox("Language", ["en", "hi"], index=0)
    auto_scroll = st.checkbox("Auto-scroll chat", value=True)
    show_debug = st.checkbox("Show debug traceback", value=False)

    st.session_state.user_settings.update(
        dict(theme=theme, language=language, auto_scroll=auto_scroll, show_debug=show_debug)
    )
    st.success("Settings saved for this session.")
