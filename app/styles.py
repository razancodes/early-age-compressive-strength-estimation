"""
Shared styles for all pages.
Import and call apply_styles() at the top of every page.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    /* Clean typography */
    .stApp, .stApp * {
        font-family: 'JetBrains Mono', monospace !important;
    }
    /* Preserve icon fonts */
    .stApp .material-symbols-rounded,
    .stApp [data-testid="stIconMaterial"],
    .stApp span[class*="icon"],
    .stApp i {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Dark mode metric cards */
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background: #1a1a2e;
            border: 1px solid #2d2d44;
        }
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1e1e1e;
        border-right: 1px solid #2d2d2d;
    }

    /* Remove default padding from top */
    .block-container { padding-top: 2rem; }

    /* Table styling */
    .stDataFrame { border-radius: 8px; }

    /* Expander (info icon) styling */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
"""


def apply_styles():
    """Inject shared CSS into the page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
