"""
Concrete Strength Predictor — Streamlit App Entry Point.

Run with: streamlit run app/app.py
"""

import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
from styles import apply_styles
apply_styles()

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏗️ Concrete Strength Predictor")
    st.caption("Stage B — Physics-Guided Stacking & Uncertainty")

    st.divider()

    st.markdown("""
    **Models (Stage B):**
    - XGBoost *(monotonic)*
    - CatBoost *(monotonic)*
    - LightGBM *(monotonic)*
    - Stacking Ensemble
    - Gaussian Process

    **Dataset:** UCI Concrete
    (1,005 samples, 30 features)

    **Validation:** 5-fold Nested CV
    """)

    st.divider()

    st.markdown("""
    **Age Subsets:**
    - **EA1** — ≤ 3 days (formwork)
    - **EA7** — ≤ 7 days (post-tension)
    - **EA14** — ≤ 14 days (QA check)
    - **Full** — 1–365 days
    """)

    st.divider()

    st.markdown("""
    **References:**
    - ASTM C39 (testing)
    - IS 456 (design)
    - ACI 211 (mix design)
    """)

    st.divider()
    st.caption("razancodes • [GitHub](https://github.com/razancodes)")

# ── Main Page ───────────────────────────────────────────────────────────
st.header("Early-Age Concrete Compressive Strength Prediction")

st.markdown(
    "Predict concrete compressive strength using physics-guided stacking ensembles "
    "and Gaussian Process regression with auto-selected models per curing age."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Stage B Models", "20")
with col2:
    st.metric("Best R²", "0.938")
with col3:
    st.metric("Best RMSE", "3.57 MPa")
with col4:
    st.metric("Dataset Samples", "1,005")

st.divider()

# ── Page Descriptions ───────────────────────────────────────────────────
st.subheader("Pages")

pages = {
    "🏗️ Full Age Prediction": (
        "Auto-selects the best model for your curing age. Shows prediction with "
        "confidence score, SHAP explanation, IS 456 grade, and standard references."
    ),
    "📈 Multi-day Prediction": (
        "Project strength development from 1 to 365 days for a mix design, "
        "with model transitions at subset boundaries and uncertainty bands."
    ),
    "🔬 Experimental Analysis": (
        "Explore all 20 Stage B models — performance overview, interactive "
        "comparison, SHAP analysis, feature engineering deep-dive, and "
        "hyperparameter details."
    ),
    "📦 Batch Predict": (
        "Upload a CSV of mix designs, run batch inference, and download "
        "predictions. Missing values auto-filled with best-practice defaults."
    ),
}

for name, desc in pages.items():
    st.markdown(f"**{name}** — {desc}")

st.divider()

# ── Quick Reference ─────────────────────────────────────────────────────
st.subheader("Quick Reference — Standard Values")

ref1, ref2, ref3 = st.columns(3)
with ref1:
    st.markdown("**Testing Standards**")
    st.markdown(
        "- Standard test age: **28 days** (ASTM C39)\n"
        "- Min curing (OPC): **7 days** (IS 456)\n"
        "- Min curing (SCM): **10 days** (IS 456)"
    )
with ref2:
    st.markdown("**Mix Design Limits**")
    st.markdown(
        "- W/B ratio: **0.35–0.65** (ACI 211)\n"
        "- W/B max (severe): **0.45** (IS 456)\n"
        "- Min RCC grade: **M20** (IS 456)"
    )
with ref3:
    st.markdown("**Strength Development**")
    st.markdown(
        "- 3-day: **33–40%** of 28-day\n"
        "- 7-day: **65–75%** of 28-day\n"
        "- Dataset range: **2.33–82.60 MPa**"
    )
