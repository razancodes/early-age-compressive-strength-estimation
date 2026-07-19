"""
Page 3: Experimental Analysis
Comprehensive view of all Stage B models, features, and training results.
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    render_input_form, engineer_features, predict, load_model,
    load_results_df, load_hyperparameters, get_plot_path, plot_exists,
    autofill_single_input, get_feature_columns,
    MODEL_TYPES, SUBSET_NAMES, MODEL_INFO, SUBSET_INFO, FEATURE_INFO,
)

apply_styles()

st.header("🔬 Experimental Analysis")
st.markdown(
    "Explore all **20 Stage B models** across 4 age subsets. Every section "
    "includes contextual explanations of models, subsets, and features."
)

# ── Tabs ────────────────────────────────────────────────────────────────
tab_perf, tab_compare, tab_shap, tab_features, tab_params = st.tabs([
    "📊 Performance", "⚖️ Compare Models", "🔍 SHAP & EDA",
    "🧬 Feature Engineering", "⚙️ Hyperparameters",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Performance Overview
# ═══════════════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("Stage B Model Performance (Nested 5-Fold CV)")

    with st.expander("ℹ️ What is Stage B?"):
        st.markdown(
            "**Stage B** applies **monotonic constraints** to all gradient boosting "
            "models, forcing predictions to obey physical laws (e.g., more cement → "
            "more strength). It also introduces **Stacking Ensembles** and **Gaussian "
            "Process** regression. Stage B dramatically outperforms unconstrained "
            "Stage A on early-age subsets."
        )

    results_df = load_results_df()
    stage_b_names = [
        "XGBoost_constrained", "CatBoost_constrained", "LightGBM_constrained",
        "Stacking_Ensemble", "GaussianProcess",
    ]
    boost_df = results_df[results_df["Model"].isin(stage_b_names)].copy()
    baseline_df = results_df[~results_df["Model"].isin(stage_b_names)].copy()

    display_cols = [
        "Subset", "Model", "RMSE_mean", "RMSE_std",
        "MAE_mean", "MAE_std", "R2_mean", "R2_std",
    ]

    st.markdown("**Stage B Constrained & Ensemble Models**")
    st.dataframe(
        boost_df[display_cols].round(4),
        width='stretch', hide_index=True,
    )

    if not baseline_df.empty:
        st.markdown("**Baseline Models (Linear Regression, Random Forest)**")
        st.dataframe(
            baseline_df[display_cols].round(4),
            width='stretch', hide_index=True,
        )

    # ── Best model per subset ──────────────────────────────────────
    st.subheader("Best Model Per Subset")

    for subset in SUBSET_NAMES:
        sdf = results_df[results_df["Subset"] == subset]
        if sdf.empty:
            continue
        best = sdf.loc[sdf["R2_mean"].idxmax()]
        si = SUBSET_INFO.get(subset, {})

        col_info, col_metrics = st.columns([1.2, 1])
        with col_info:
            st.markdown(f"**{si.get('title', subset)}**")
            st.caption(si.get("description", ""))
        with col_metrics:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Best Model", str(best["Model"])[:20])
            with mc2:
                st.metric("R²", f"{best['R2_mean']:.4f}")
            with mc3:
                st.metric("RMSE", f"{best['RMSE_mean']:.3f} MPa")
        st.divider()

    # ── Heatmap & comparison plots ─────────────────────────────────
    st.subheader("Visual Comparison")
    vc1, vc2 = st.columns(2)
    with vc1:
        path = get_plot_path("r2_heatmap")
        if plot_exists(path):
            st.image(path, caption="R² Heatmap", width='stretch')
    with vc2:
        path = get_plot_path("model_comparison")
        if plot_exists(path):
            st.image(path, caption="Model Comparison", width='stretch')

    # ── Model Architecture Guide ───────────────────────────────────
    st.subheader("Model Architecture Guide")
    for key, info in MODEL_INFO.items():
        with st.expander(f"ℹ️ {info['title']}"):
            st.markdown(info["description"])


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Interactive Model Comparison
# ═══════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare All 20 Models on One Input")

    with st.expander("ℹ️ How Model Comparison Works"):
        st.markdown(
            "Enter a mix design to run **all 20 Stage B model variants** "
            "(5 models × 4 subsets) on the same input. High agreement "
            "(low CV%) means the input falls within the training distribution."
        )

    left_cmp, right_cmp = st.columns([1, 1.6])

    with left_cmp:
        st.subheader("Mix Design Input")
        raw_input_cmp = render_input_form(include_age=True, key_prefix="cmp_")
        run_cmp = st.button(
            "Compare All Models", type="primary", width='stretch',
        )

    with right_cmp:
        if run_cmp:
            filled_cmp, fill_log_cmp = autofill_single_input(raw_input_cmp)
            if fill_log_cmp:
                st.info(
                    "**Auto-filled:**\n- " + "\n- ".join(fill_log_cmp)
                )

            features_cmp = engineer_features(filled_cmp)

            # Run all 20 predictions
            predictions = {}
            for mt in MODEL_TYPES:
                predictions[mt] = {}
                for subset in SUBSET_NAMES:
                    try:
                        model = load_model(mt, subset)
                        if mt == "GP":
                            from src.gp_model import gp_predict_with_uncertainty
                            X_cols = get_feature_columns()
                            X = features_cmp[X_cols].values
                            mean_p, _ = gp_predict_with_uncertainty(model, X)
                            predictions[mt][subset] = round(float(mean_p[0]), 2)
                        else:
                            predictions[mt][subset] = round(
                                predict(model, features_cmp), 2,
                            )
                    except Exception:
                        predictions[mt][subset] = None

            # ── Predictions table ──────────────────────────────────
            st.subheader("Predictions (MPa)")
            table_data = []
            for mt in MODEL_TYPES:
                row = {"Model": mt}
                for subset in SUBSET_NAMES:
                    val = predictions[mt][subset]
                    row[subset] = val if val is not None else "—"
                table_data.append(row)
            st.dataframe(
                pd.DataFrame(table_data),
                width='stretch', hide_index=True,
            )

            # ── Agreement statistics ───────────────────────────────
            st.subheader("Prediction Agreement")
            with st.expander("ℹ️ About Prediction Agreement"):
                st.markdown(
                    "**CV (Coefficient of Variation)** = Std / Mean × 100%. "
                    "CV < 5% = excellent agreement; 5–15% = moderate; > 15% = "
                    "the input may lie outside the training distribution."
                )
            all_vals = [
                v for mt in predictions
                for v in predictions[mt].values()
                if v is not None
            ]
            if all_vals:
                arr = np.array(all_vals)
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    st.metric("Mean", f"{arr.mean():.2f} MPa")
                with ac2:
                    st.metric("Std Dev", f"{arr.std():.2f} MPa")
                with ac3:
                    cv = arr.std() / arr.mean() * 100 if arr.mean() else 0
                    st.metric("CV", f"{cv:.1f}%")

            # ── Bar chart ──────────────────────────────────────────
            st.subheader("Comparison Chart")
            colors = {
                "XGBoost": "#2196F3", "CatBoost": "#FF9800",
                "LightGBM": "#4CAF50", "Stacking": "#9C27B0",
                "GP": "#E64A19",
            }
            fig = go.Figure()
            for mt in MODEL_TYPES:
                vals = [predictions[mt].get(s) for s in SUBSET_NAMES]
                fig.add_trace(go.Bar(
                    name=mt, x=SUBSET_NAMES, y=vals,
                    marker_color=colors.get(mt, "#666"),
                ))
            fig.update_layout(
                barmode="group", xaxis_title="Subset",
                yaxis_title="Predicted Strength (MPa)", height=400,
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                template="plotly_white",
            )
            st.plotly_chart(fig, width='stretch')

            # ── Training R² reference ──────────────────────────────
            st.subheader("Training R² Reference")
            with st.expander("ℹ️ About R²"):
                st.markdown(
                    "R² measures explained variance. 1.0 = perfect fit. "
                    "Values are from **nested 5-fold CV** — an unbiased "
                    "estimate of generalization performance."
                )
            MODEL_NAME_MAP = {
                "XGBoost": "XGBoost_constrained",
                "CatBoost": "CatBoost_constrained",
                "LightGBM": "LightGBM_constrained",
                "Stacking": "Stacking_Ensemble",
                "GP": "GaussianProcess",
            }
            results_ref = load_results_df()
            r2_data = []
            for mt in MODEL_TYPES:
                row = {"Model": mt}
                for subset in SUBSET_NAMES:
                    match = results_ref[
                        (results_ref["Model"] == MODEL_NAME_MAP.get(mt, mt))
                        & (results_ref["Subset"] == subset)
                    ]
                    row[subset] = (
                        f"{match['R2_mean'].iloc[0]:.4f}"
                        if not match.empty else "—"
                    )
                r2_data.append(row)
            st.dataframe(
                pd.DataFrame(r2_data),
                width='stretch', hide_index=True,
            )
        else:
            st.info("Enter a mix design and click **Compare All Models**.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: SHAP & Explainability
# ═══════════════════════════════════════════════════════════════════════
with tab_shap:
    st.subheader("SHAP Feature Importance & Summary")

    with st.expander("ℹ️ About SHAP Analysis"):
        st.markdown(
            "**SHAP (SHapley Additive exPlanations)** uses game theory to assign "
            "each feature a contribution score for every prediction:\n\n"
            "- **Importance**: Mean |SHAP| per feature (global ranking)\n"
            "- **Beeswarm**: Distribution of SHAP values, colored by feature value"
        )

    c1s, c2s = st.columns(2)
    with c1s:
        shap_model_sel = st.selectbox(
            "Model", ["XGBoost", "CatBoost", "LightGBM"], key="shap_model",
        )
    with c2s:
        shap_subset_sel = st.selectbox(
            "Subset", SUBSET_NAMES, key="shap_subset",
        )

    # Subset context
    si = SUBSET_INFO.get(shap_subset_sel, {})
    with st.expander(f"ℹ️ {si.get('title', shap_subset_sel)}"):
        st.markdown(si.get("description", ""))

    col_imp, col_sum = st.columns(2)
    with col_imp:
        path = get_plot_path("shap_importance", shap_model_sel, shap_subset_sel)
        if plot_exists(path):
            st.image(
                path, caption="Feature Importance (mean |SHAP|)",
                width='stretch',
            )
        else:
            st.warning("SHAP importance plot not found.")
    with col_sum:
        path = get_plot_path("shap_summary", shap_model_sel, shap_subset_sel)
        if plot_exists(path):
            st.image(
                path, caption="Beeswarm Summary",
                width='stretch',
            )
        else:
            st.warning("SHAP summary plot not found.")

    # ── Dependence plots ───────────────────────────────────────────
    st.subheader("SHAP Dependence Plots")
    with st.expander("ℹ️ About Dependence Plots"):
        st.markdown(
            "Shows how a **single feature** affects predictions across the "
            "dataset. Each dot is a sample; the vertical axis shows the SHAP "
            "contribution of that feature."
        )

    dep_candidates = [
        "W_B_ratio", "Cement", "Binder", "Age", "log_Age",
        "Blast_Furnace_Slag", "Fly_Ash", "Water",
        "Superplasticizer", "Total_Aggregate",
    ]
    dep_features = [
        f for f in dep_candidates
        if plot_exists(
            get_plot_path("shap_dep", shap_model_sel, shap_subset_sel, feature=f)
        )
    ]

    if dep_features:
        cols_dep = st.columns(min(3, len(dep_features)))
        for i, feat in enumerate(dep_features[:3]):
            with cols_dep[i]:
                path = get_plot_path(
                    "shap_dep", shap_model_sel, shap_subset_sel, feature=feat,
                )
                st.image(path, caption=feat, width='stretch')
                fi = FEATURE_INFO.get(feat, {})
                st.caption(fi.get("description", "")[:150])

        if len(dep_features) > 3:
            cols_dep2 = st.columns(min(3, len(dep_features) - 3))
            for i, feat in enumerate(dep_features[3:6]):
                with cols_dep2[i]:
                    path = get_plot_path(
                        "shap_dep", shap_model_sel, shap_subset_sel,
                        feature=feat,
                    )
                    st.image(path, caption=feat, width='stretch')
    else:
        st.info("No dependence plots found for this model/subset.")

    # ── EDA ────────────────────────────────────────────────────────
    st.subheader("Exploratory Data Analysis")
    with st.expander("ℹ️ About EDA Plots"):
        st.markdown(
            "Distribution and correlation analysis of the UCI Concrete Dataset "
            "(1,005 samples). Helps understand the data landscape."
        )

    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        path = get_plot_path("eda_target_dist")
        if plot_exists(path):
            st.image(
                path, caption="Target Distribution",
                width='stretch',
            )
    with eda_c2:
        path = get_plot_path("eda_strength_vs_age")
        if plot_exists(path):
            st.image(
                path, caption="Strength vs Curing Age",
                width='stretch',
            )

    path = get_plot_path("eda_correlation")
    if plot_exists(path):
        st.image(
            path, caption="Feature Correlation Matrix",
            width='stretch',
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Feature Engineering Deep-Dive
# ═══════════════════════════════════════════════════════════════════════
with tab_features:
    st.subheader("30-Feature Engineering Pipeline")

    with st.expander("ℹ️ Feature Engineering Overview"):
        st.markdown(
            "8 raw inputs are expanded into **30 features** using concrete "
            "engineering domain knowledge (ACI 211, IS 10262, Powers' theory). "
            "Stage A adds 14 features (binder, aggregates, temporal). "
            "Stage B adds 8 more (physics ratios, interaction terms)."
        )

    # Group features by category
    groups = {}
    for fname, finfo in FEATURE_INFO.items():
        g = finfo.get("group", "Other")
        groups.setdefault(g, []).append((fname, finfo))

    for group_name, features in groups.items():
        st.markdown(f"### {group_name}")
        for fname, finfo in features:
            with st.expander(
                f"ℹ️ **{fname}** — `{finfo.get('formula', '')}`"
            ):
                st.markdown(finfo.get("description", ""))
        st.markdown("")


# ═══════════════════════════════════════════════════════════════════════
# TAB 5: Hyperparameters
# ═══════════════════════════════════════════════════════════════════════
with tab_params:
    st.subheader("Optimized Hyperparameters")

    with st.expander("ℹ️ Hyperparameter Optimization"):
        st.markdown(
            "Hyperparameters found via **Optuna TPE** (Tree-structured Parzen "
            "Estimator) with **100 trials** per model-subset inside the inner "
            "loop of nested 5-fold CV. The test fold never influences "
            "hyperparameter choices."
        )

    params = load_hyperparameters()

    hp_explanations = {
        "XGBoost": (
            "- **max_depth**: Maximum tree depth (complexity control)\n"
            "- **learning_rate**: Step size shrinkage (lower = robust)\n"
            "- **n_estimators**: Number of boosting rounds\n"
            "- **subsample**: Row sampling ratio per tree\n"
            "- **colsample_bytree**: Column sampling ratio\n"
            "- **reg_alpha/lambda**: L1/L2 regularization"
        ),
        "CatBoost": (
            "- **depth**: Tree depth (symmetric trees)\n"
            "- **learning_rate**: Step size shrinkage\n"
            "- **iterations**: Number of boosting iterations\n"
            "- **l2_leaf_reg**: L2 regularization on leaves\n"
            "- **bagging_temperature**: Bootstrap intensity\n"
            "- **random_strength**: Score randomization magnitude"
        ),
        "LightGBM": (
            "- **num_leaves**: Max leaves (leaf-wise growth)\n"
            "- **learning_rate**: Step size shrinkage\n"
            "- **n_estimators**: Number of boosting rounds\n"
            "- **subsample**: Row sampling ratio\n"
            "- **colsample_bytree**: Column sampling ratio\n"
            "- **reg_alpha/lambda**: L1/L2 regularization\n"
            "- **min_child_samples**: Min samples per leaf"
        ),
    }

    for model_type_hp in ["XGBoost", "CatBoost", "LightGBM"]:
        mi = MODEL_INFO.get(model_type_hp, {})
        st.markdown(f"**{mi.get('title', model_type_hp)}**")

        with st.expander(f"ℹ️ {model_type_hp} Hyperparameters Explained"):
            st.markdown(hp_explanations.get(model_type_hp, ""))

        param_rows = []
        for subset in SUBSET_NAMES:
            key = f"{model_type_hp}_{subset}"
            if key in params:
                row = {"Subset": subset}
                row.update(params[key])
                param_rows.append(row)
        if param_rows:
            st.dataframe(
                pd.DataFrame(param_rows),
                width='stretch', hide_index=True,
            )
        st.markdown("")
