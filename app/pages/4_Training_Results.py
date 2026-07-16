"""
Page 4: Training Results — Dashboard of all training metrics and plots.
"""

import sys
import os

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    load_results_df, load_hyperparameters, get_plot_path, plot_exists,
    MODEL_TYPES, SUBSET_NAMES,
)

apply_styles()

st.header("Training Results")
st.markdown("Browse all training metrics, prediction plots, residuals, "
            "and SHAP analysis from the Stage B pipeline.")

# ── Tab Layout ──────────────────────────────────────────────────────────
tab_overview, tab_plots, tab_shap, tab_eda, tab_params = st.tabs([
    "Overview", "Prediction Plots", "SHAP Analysis", "EDA", "Hyperparameters"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB: Overview
# ═══════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Performance Summary")

    results_df = load_results_df()
    stage_b_models = ['XGBoost_constrained', 'CatBoost_constrained', 'LightGBM_constrained', 'Stacking_Ensemble', 'GaussianProcess']
    boost_df = results_df[results_df["Model"].isin(stage_b_models)].copy()
    baseline_df = results_df[~results_df["Model"].isin(stage_b_models)].copy()

    # Format for display
    display_cols = ["Subset", "Model", "RMSE_mean", "RMSE_std",
                    "MAE_mean", "MAE_std", "R2_mean", "R2_std"]

    st.markdown("**Stage B Constrained & Ensemble Models (Nested CV)**")
    st.dataframe(
        boost_df[display_cols].round(4),
        use_container_width=True, hide_index=True,
    )

    st.markdown("**Baseline Models (5-Fold CV)**")
    st.dataframe(
        baseline_df[display_cols].round(4),
        use_container_width=True, hide_index=True,
    )

    # Best per subset
    st.subheader("Best Model Per Subset")
    best_data = []
    for subset in SUBSET_NAMES:
        sdf = results_df[results_df["Subset"] == subset]
        if not sdf.empty:
            best = sdf.loc[sdf["R2_mean"].idxmax()]
            best_data.append({
                "Subset": subset,
                "Best Model": best["Model"],
                "R-squared": round(best["R2_mean"], 4),
                "RMSE (MPa)": round(best["RMSE_mean"], 3),
            })
    st.dataframe(pd.DataFrame(best_data), use_container_width=True,
                 hide_index=True)

    # Heatmap and comparison
    st.subheader("Visual Comparison")
    c1, c2 = st.columns(2)
    with c1:
        path = get_plot_path("r2_heatmap")
        if plot_exists(path):
            st.image(path, caption="R-squared Heatmap", use_container_width=True)
    with c2:
        path = get_plot_path("model_comparison")
        if plot_exists(path):
            st.image(path, caption="Model Comparison", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB: Prediction Plots
# ═══════════════════════════════════════════════════════════════════════
with tab_plots:
    st.subheader("Prediction Scatter and Residual Plots")

    c1, c2 = st.columns(2)
    with c1:
        sel_model = st.selectbox("Model", ["XGBoost", "CatBoost", "LightGBM", "LinearRegression", "RandomForest"],
                                 key="plots_model")
    with c2:
        sel_subset = st.selectbox("Subset", SUBSET_NAMES, key="plots_subset")

    col_scatter, col_resid = st.columns(2)

    with col_scatter:
        path = get_plot_path("pred_scatter", sel_model, sel_subset)
        if plot_exists(path):
            st.image(path, caption=f"Predicted vs Measured -- {sel_model} {sel_subset}",
                     use_container_width=True)
        else:
            st.warning(f"Plot not found: pred_scatter_{sel_model}_{sel_subset}")

    with col_resid:
        path = get_plot_path("residuals", sel_model, sel_subset)
        if plot_exists(path):
            st.image(path, caption=f"Residuals -- {sel_model} {sel_subset}",
                     use_container_width=True)
        else:
            st.warning(f"Plot not found: residuals_{sel_model}_{sel_subset}")


# ═══════════════════════════════════════════════════════════════════════
# TAB: SHAP Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab_shap:
    st.subheader("SHAP Feature Importance and Summary")

    c1, c2 = st.columns(2)
    with c1:
        shap_model = st.selectbox("Model", ["XGBoost", "CatBoost", "LightGBM"], key="shap_model")
    with c2:
        shap_subset = st.selectbox("Subset", SUBSET_NAMES, key="shap_subset")

    # Importance and summary side by side
    col_imp, col_sum = st.columns(2)

    with col_imp:
        path = get_plot_path("shap_importance", shap_model, shap_subset)
        if plot_exists(path):
            st.image(path, caption="Feature Importance (mean |SHAP|)",
                     use_container_width=True)
        else:
            st.warning("SHAP importance plot not found.")

    with col_sum:
        path = get_plot_path("shap_summary", shap_model, shap_subset)
        if plot_exists(path):
            st.image(path, caption="Beeswarm Summary",
                     use_container_width=True)
        else:
            st.warning("SHAP summary plot not found.")

    # Dependence plots
    st.subheader("SHAP Dependence Plots")
    st.caption("Shows how a single feature affects predictions across the dataset.")

    # Find available dependence plots for this model/subset
    dep_features = []
    for candidate in ["W_B_ratio", "Cement", "Binder", "Age", "log_Age",
                       "Blast_Furnace_Slag", "Fly_Ash", "Water",
                       "Superplasticizer", "Total_Aggregate"]:
        path = get_plot_path("shap_dep", shap_model, shap_subset, feature=candidate)
        if plot_exists(path):
            dep_features.append(candidate)

    if dep_features:
        cols = st.columns(min(3, len(dep_features)))
        for i, feat in enumerate(dep_features[:3]):
            with cols[i]:
                path = get_plot_path("shap_dep", shap_model, shap_subset, feature=feat)
                st.image(path, caption=feat, use_container_width=True)

        # Show remaining if more than 3
        if len(dep_features) > 3:
            cols2 = st.columns(min(3, len(dep_features) - 3))
            for i, feat in enumerate(dep_features[3:6]):
                with cols2[i]:
                    path = get_plot_path("shap_dep", shap_model, shap_subset,
                                         feature=feat)
                    st.image(path, caption=feat, use_container_width=True)
    else:
        st.info("No dependence plots found for this model/subset combination.")


# ═══════════════════════════════════════════════════════════════════════
# TAB: EDA
# ═══════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        path = get_plot_path("eda_target_dist")
        if plot_exists(path):
            st.image(path, caption="Target Distribution",
                     use_container_width=True)

    with c2:
        path = get_plot_path("eda_strength_vs_age")
        if plot_exists(path):
            st.image(path, caption="Strength vs Curing Age",
                     use_container_width=True)

    path = get_plot_path("eda_correlation")
    if plot_exists(path):
        st.image(path, caption="Feature Correlation Matrix",
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB: Hyperparameters
# ═══════════════════════════════════════════════════════════════════════
with tab_params:
    st.subheader("Optimized Hyperparameters")
    st.caption("Found via Optuna TPE sampler with 100 trials per model-subset.")

    params = load_hyperparameters()

    for model_type in ["XGBoost", "CatBoost", "LightGBM"]:
        st.markdown(f"**{model_type}**")
        param_rows = []
        for subset in SUBSET_NAMES:
            key = f"{model_type}_{subset}"
            if key in params:
                row = {"Subset": subset}
                row.update(params[key])
                param_rows.append(row)
        if param_rows:
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True,
                         hide_index=True)
        st.markdown("")  # spacing
