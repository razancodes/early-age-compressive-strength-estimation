"""
Page 1: Predict — Single mix prediction with SHAP explanation.
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    render_input_form, engineer_features, predict, get_shap_explanation,
    load_model, get_feature_columns, autofill_single_input,
    MODEL_TYPES, SUBSET_NAMES, DISPLAY_NAMES,
)

apply_styles()

st.header("Predict Compressive Strength")
st.markdown("Enter a concrete mix design to predict compressive strength. "
            "The SHAP waterfall plot shows which features drove the prediction. "
            "Check N/A for any unavailable value -- it will be auto-filled.")

# ── Layout ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.4])

with left:
    st.subheader("Mix Design Input")
    raw_input = render_input_form(include_age=True, key_prefix="pred_")

    st.divider()

    col_m, col_s = st.columns(2)
    with col_m:
        model_type = st.selectbox("Model", MODEL_TYPES, index=1, key="pred_model")
    with col_s:
        subset = st.selectbox("Subset", SUBSET_NAMES, index=3, key="pred_subset")

    run = st.button("Predict", type="primary", use_container_width=True)

with right:
    if run:
        # Auto-fill any N/A values
        filled_input, fill_log = autofill_single_input(raw_input)
        if fill_log:
            st.info("**Auto-filled values:**\n- " + "\n- ".join(fill_log))

        # Engineer features
        features_df = engineer_features(filled_input)
        feature_cols = get_feature_columns()

        # Load model and predict
        model = load_model(model_type, subset)
        
        if model_type == "GP":
            from src.gp_model import gp_predict_with_uncertainty
            X_cols = get_feature_columns()
            X = features_df[X_cols].values
            mean, std = gp_predict_with_uncertainty(model, X)
            strength = float(mean[0])
            uncertainty_std = float(std[0])
        else:
            strength = predict(model, features_df)

        # ── Result ──────────────────────────────────────────────────
        st.subheader("Prediction Result")
        if model_type == "GP":
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.metric("Predicted Compressive Strength", f"{strength:.2f} MPa")
            with c_res2:
                st.metric("Uncertainty (±1 Std Dev)", f"{uncertainty_std:.2f} MPa")
            
            # 90% prediction interval (mean +- 1.645 * std)
            lower_bound = max(0.0, strength - 1.645 * uncertainty_std)
            upper_bound = strength + 1.645 * uncertainty_std
            st.markdown(f"**90% Prediction Interval:** `{lower_bound:.2f}` to `{upper_bound:.2f}` MPa")
        else:
            st.metric("Predicted Compressive Strength", f"{strength:.2f} MPa")

        # ── Derived Features ────────────────────────────────────────
        st.subheader("Derived Features")
        derived = {
            "Binder (kg/m3)": f"{features_df['Binder'].iloc[0]:.1f}",
            "W/B Ratio": f"{features_df['W_B_ratio'].iloc[0]:.3f}",
            "GGBS Ratio": f"{features_df['GGBS_ratio'].iloc[0]:.3f}",
            "Fly Ash Ratio": f"{features_df['FlyAsh_ratio'].iloc[0]:.3f}",
            "SCM Ratio": f"{features_df['SCM_ratio'].iloc[0]:.3f}",
            "Total Aggregate (kg/m3)": f"{features_df['Total_Aggregate'].iloc[0]:.1f}",
            "Fine Agg Ratio": f"{features_df['Fine_Agg_ratio'].iloc[0]:.3f}",
            "Agg/Binder Ratio": f"{features_df['Agg_Binder_ratio'].iloc[0]:.3f}",
            "SP per Binder": f"{features_df['SP_per_binder'].iloc[0]:.4f}",
            "log(Age)": f"{features_df['log_Age'].iloc[0]:.3f}",
        }
        cols = st.columns(2)
        for i, (k, v) in enumerate(derived.items()):
            with cols[i % 2]:
                st.markdown(f"**{k}:** {v}")

        # ── SHAP Waterfall ──────────────────────────────────────────
        st.subheader("SHAP Explanation")
        st.caption(f"Feature contributions to this prediction ({model_type} {subset})")

        try:
            explanation = get_shap_explanation(model, features_df)
            fig, ax = plt.subplots(figsize=(8, 6))
            shap_plot = shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP plot: {e}")

    else:
        st.info("Configure the mix design on the left and click Predict.")
