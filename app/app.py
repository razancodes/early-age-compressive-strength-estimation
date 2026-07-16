"""
Concrete Strength Predictor — Streamlit App Entry Point.

Run with: streamlit run app/app.py
"""

import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
from styles import apply_styles
apply_styles()

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Concrete Strength Predictor")
    st.caption("Stage B — Physics-Guided Stacking & Uncertainty")
    st.markdown("[View on GitHub](https://github.com/razancodes/compressive-strength-estimation)")

    st.divider()

    st.markdown("""
    **Models:** XGBoost, CatBoost, LightGBM, Stacking, Gaussian Process

    **Trained on:** UCI Concrete Dataset
    (1,030 samples, 30 features)

    **Validation:** 5-fold Nested CV
    with Monotonic Constraints
    """)

    st.divider()

    st.markdown("""
    **Subsets:**
    - **EA1** — up to 3 days
    - **EA7** — up to 7 days
    - **EA14** — up to 14 days
    - **Full** — all ages (1-365 days)
    """)

    st.divider()

    st.markdown("""
    **razancodes**
    [GitHub](https://github.com/razancodes)
    [razancodes@gmail.com](mailto:razancodes@gmail.com)
    """, unsafe_allow_html=True)

# ── Main Page ───────────────────────────────────────────────────────────
st.header("Early-Age Concrete Compressive Strength Prediction")

st.markdown("""
Predict the compressive strength of concrete mixes using physics-guided stacking ensembles
and Gaussian Process regression models with conformal uncertainty intervals.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Trained Models", "20")
with col2:
    st.metric("Best R-squared", "0.956")
with col3:
    st.metric("Best RMSE", "3.10 MPa")
with col4:
    st.metric("Training Samples", "1,030")

st.divider()

st.subheader("Pages")

pages = {
    "Predict": "Enter a mix design and get an instant strength prediction, conformal uncertainty interval, and SHAP explanation.",
    "Strength Curve": "Project strength development across ages 1 to 365 days for a single mix, with uncertainty bounds.",
    "Compare Models": "Run all 20 model variants on the same input and compare predictions.",
    "Training Results": "Browse all training metrics, scatter plots, residuals, and SHAP analysis.",
    "Batch Predict": "Upload a CSV of mix designs and download predictions.",
}

for name, desc in pages.items():
    st.markdown(f"**{name}** — {desc}")
