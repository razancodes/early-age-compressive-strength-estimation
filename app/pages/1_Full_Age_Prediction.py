"""
Page 1: Predict Compressive Strength
Auto-selects the best model per age bracket with confidence analysis and SHAP.
"""

import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    render_input_form, engineer_features, predict, get_shap_explanation,
    load_model, get_feature_columns, autofill_single_input,
    get_best_model_for_age, compute_model_agreement, classify_grade,
    SHAP_FALLBACK, SUBSET_INFO, MODEL_INFO, FEATURE_INFO,
)

apply_styles()

st.header("🏗️ Full Age Prediction")
st.markdown(
    "Enter a concrete mix design — the app **auto-selects** the best-performing "
    "model for your curing age. Includes confidence analysis, SHAP explainability, "
    "and IS 456 grade classification."
)

# ── Layout ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.5])

with left:
    st.subheader("Mix Design Input")
    raw_input = render_input_form(include_age=True, key_prefix="pred_")
    run = st.button("🔬 Predict Strength", type="primary", width='stretch')

with right:
    if run:
        # Auto-fill missing values
        filled_input, fill_log = autofill_single_input(raw_input)
        if fill_log:
            st.info("**Auto-filled values:**\n- " + "\n- ".join(fill_log))

        age = filled_input["Age"]
        best = get_best_model_for_age(age)
        subset = best["subset"]
        model_type = best["model_type"]

        # ── Model & Subset explanation ──────────────────────────────
        with st.expander(
            f"ℹ️ Using {best['display']} on {subset} subset", expanded=False
        ):
            si = SUBSET_INFO[subset]
            mi = MODEL_INFO[model_type]
            st.markdown(f"**Subset: {si['title']}**")
            st.markdown(si["description"])
            st.divider()
            st.markdown(f"**Model: {mi['title']}**")
            st.markdown(mi["description"])
            st.caption(
                f"Training R² = {best['r2']:.4f} | RMSE = {best['rmse']:.3f} MPa "
                f"| {si['samples']} samples | {si['construction_use']}"
            )

        # ── Prediction ─────────────────────────────────────────────
        features_df = engineer_features(filled_input)
        model = load_model(model_type, subset)

        gp_std = None
        if model_type == "GP":
            from src.gp_model import gp_predict_with_uncertainty
            X_cols = get_feature_columns()
            X = features_df[X_cols].values
            mean_pred, std_pred = gp_predict_with_uncertainty(model, X)
            strength = float(mean_pred[0])
            gp_std = float(std_pred[0])
        else:
            strength = predict(model, features_df)

        # ── Model Agreement ────────────────────────────────────────
        agreement = compute_model_agreement(features_df, subset)

        # ── GP fallback uncertainty ────────────────────────────────
        if gp_std is None:
            try:
                gp_model = load_model("GP", subset)
                from src.gp_model import gp_predict_with_uncertainty
                X_cols = get_feature_columns()
                X = features_df[X_cols].values
                _, gp_s = gp_predict_with_uncertainty(gp_model, X)
                gp_std = float(gp_s[0])
            except Exception:
                gp_std = None

        # ── Grade Classification ───────────────────────────────────
        grade, grade_desc, is_structural = classify_grade(strength)

        # ── Results Display ────────────────────────────────────────
        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted Strength", f"{strength:.2f} MPa")
        with c2:
            st.metric("Concrete Grade", grade)
        with c3:
            if agreement["cv_percent"] is not None:
                cv = agreement["cv_percent"]
                conf = "High" if cv < 5 else "Moderate" if cv < 15 else "Low"
                st.metric(
                    "Model Agreement", conf,
                    delta=f"CV = {cv:.1f}%", delta_color="inverse",
                )
            else:
                st.metric("Model Agreement", "N/A")

        st.caption(f"**{grade}** — {grade_desc}")

        # ── Confidence & Uncertainty Details ───────────────────────
        with st.expander("ℹ️ Confidence & Uncertainty Details"):
            st.markdown(
                "**Model Agreement** — Three constrained tree models "
                "(XGBoost, CatBoost, LightGBM) predict on the same input. "
                "Low CV% indicates the prediction is robust across architectures."
            )
            if agreement["predictions"]:
                cols_agr = st.columns(len(agreement["predictions"]))
                for i, (mt, pred) in enumerate(agreement["predictions"].items()):
                    with cols_agr[i]:
                        st.metric(mt, f"{pred:.2f} MPa")
                st.caption(
                    f"Mean: {agreement['mean']:.2f} MPa | "
                    f"Std: {agreement['std']:.2f} MPa | "
                    f"CV: {agreement['cv_percent']:.1f}%"
                )

            if gp_std is not None:
                st.divider()
                st.markdown("**Gaussian Process Uncertainty**")
                lower = max(0.0, strength - 1.645 * gp_std)
                upper = strength + 1.645 * gp_std
                c_gp1, c_gp2 = st.columns(2)
                with c_gp1:
                    st.metric("GP Std Dev", f"±{gp_std:.2f} MPa")
                with c_gp2:
                    st.metric("90% Interval", f"{lower:.1f} – {upper:.1f} MPa")

        # ── Standards Reference ────────────────────────────────────
        st.subheader("Standards Reference")
        wb = features_df["W_B_ratio"].iloc[0]

        r1, r2, r3 = st.columns(3)
        with r1:
            wb_ok = 0.35 <= wb <= 0.65
            st.metric("W/B Ratio", f"{wb:.3f}")
            st.caption(
                f"{'✅ Within' if wb_ok else '⚠️ Outside'} ACI 211 range (0.35–0.65)"
            )
        with r2:
            struct = "✅ Structural" if is_structural else "⚠️ Non-structural"
            st.metric("IS 456 Classification", struct)
            st.caption("Min M20 required for RCC")
        with r3:
            st.metric("Model R²", f"{best['r2']:.4f}")
            st.caption(f"{best['display']} on {subset}")

        # ── Derived Features ───────────────────────────────────────
        with st.expander("📐 Derived Features (30 engineered)", expanded=False):
            feature_cols = get_feature_columns()
            feat_rows = []
            for col in feature_cols:
                val = features_df[col].iloc[0]
                info = FEATURE_INFO.get(col, {})
                feat_rows.append({
                    "Feature": col,
                    "Value": f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}",
                    "Group": info.get("group", "—"),
                    "Formula": info.get("formula", "—"),
                })
            st.dataframe(
                pd.DataFrame(feat_rows),
                width='stretch', hide_index=True,
            )

            # Per-group feature explanations
            groups = {}
            for col in feature_cols:
                info = FEATURE_INFO.get(col, {})
                g = info.get("group", "Other")
                groups.setdefault(g, []).append(col)

            for group, cols in groups.items():
                with st.expander(f"ℹ️ {group}"):
                    for col in cols:
                        info = FEATURE_INFO.get(col, {})
                        st.markdown(
                            f"**{col}** — `{info.get('formula', 'N/A')}`"
                        )
                        st.caption(info.get("description", ""))

        # ── SHAP Explanation ───────────────────────────────────────
        st.subheader("SHAP Explanation (xAI)")

        shap_info = SHAP_FALLBACK[subset]
        is_tree = model_type in ("XGBoost", "CatBoost", "LightGBM")
        shap_model_type = model_type if is_tree else shap_info["model_type"]
        is_fallback = not is_tree

        with st.expander("ℹ️ About SHAP Explanations"):
            st.markdown(
                "**SHAP** decomposes the prediction into feature contributions. "
                "Each bar shows how much a feature pushed the prediction above "
                "(red) or below (blue) the model's baseline output."
            )
            if is_fallback:
                st.markdown(
                    f"Since {model_type} does not support tree-based SHAP, "
                    f"explanations are computed using **{shap_model_type}** "
                    f"(R² = {shap_info['r2']:.4f}), the best tree model "
                    f"for {subset}."
                )

        try:
            shap_model = load_model(shap_model_type, subset)
            explanation = get_shap_explanation(shap_model, features_df)

            if is_fallback:
                st.caption(
                    f"SHAP via {shap_model_type} "
                    f"(R²={shap_info['r2']:.4f}) — "
                    f"{model_type} does not support TreeSHAP"
                )

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP plot: {e}")

    else:
        st.info(
            "Configure the mix design on the left and click "
            "**Predict Strength**."
        )
