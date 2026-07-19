"""
Page 2: Strength Curve — Multi-age strength development projection.
Auto-selects models across subsets to plot a continuous curve.
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
    select_subset_for_age, autofill_single_input, get_feature_columns,
    MODEL_TYPES, get_best_model_for_age,
)

apply_styles()

st.header("📈 Multi-day Prediction")
st.markdown(
    "Enter a mix design to project strength development from 1 to 365 days. "
    "The app **auto-selects** the best model for each age bracket, displaying "
    "a continuous curve with model transitions."
)

with st.expander("ℹ️ How the Strength Curve Works"):
    st.markdown(
        "To plot a continuous strength curve, the app uses different models "
        "for different age brackets:\n\n"
        "- **1–3 days:** XGBoost (EA1)\n"
        "- **4–7 days:** Gaussian Process (EA7)\n"
        "- **8–14 days:** Stacking Ensemble (EA14)\n"
        "- **15–365 days:** Stacking Ensemble (Full)\n\n"
        "Gaussian Process predictions (days 4-7) include 90% confidence bands."
    )

# ── Input ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6])

with left:
    st.subheader("Mix Design")
    raw_input = render_input_form(include_age=False, key_prefix="curve_")

    st.divider()

    plot_mode = st.radio(
        "Plot Mode",
        ["Auto-select best models", "Force single architecture", "Overlay all architectures"],
        help="Choose how the strength curve is generated."
    )
    if plot_mode == "Force single architecture":
        model_type = st.selectbox("Model", MODEL_TYPES, index=1, key="curve_model")

    # Age points to predict
    ages = [1, 3, 7, 14, 28, 56, 90, 180, 365]

    run = st.button("Generate Curve", type="primary", width='stretch')

with right:
    if run:
        # Auto-fill missing values
        filled_input, fill_log = autofill_single_input(raw_input)
        if fill_log:
            st.info("**Auto-filled values:**\n- " + "\n- ".join(fill_log))

        fig = go.Figure()
        results_table = []
        
        if plot_mode == "Auto-select best models":
            # Auto-select best model for each age
            predictions = []
            stds = []
            models_used = []
            
            for age in ages:
                inp = dict(filled_input)
                inp["Age"] = float(age)
                best = get_best_model_for_age(age)
                mt = best["model_type"]
                subset = best["subset"]
                models_used.append(f"{mt} ({subset})")
                
                model = load_model(mt, subset)
                features_df = engineer_features(inp)
                
                if mt == "GP":
                    from src.gp_model import gp_predict_with_uncertainty
                    X_cols = get_feature_columns()
                    X = features_df[X_cols].values
                    mean_p, std_p = gp_predict_with_uncertainty(model, X)
                    strength = float(mean_p[0])
                    predictions.append(strength)
                    stds.append(float(std_p[0]))
                else:
                    strength = predict(model, features_df)
                    predictions.append(strength)
                    stds.append(0.0)
                    
                results_table.append({
                    "Age (days)": age,
                    "Subset": subset,
                    "Best Model": mt,
                    "Predicted Strength (MPa)": round(strength, 2)
                })
                
            fig.add_trace(go.Scatter(
                x=ages, y=predictions, mode="lines+markers",
                name="Auto-Selected Best Model",
                line=dict(color="#2196F3", width=2), marker=dict(size=8),
            ))
            
            # Confidence bands for GP points
            if any(s > 0 for s in stds):
                lower_bounds = [max(0.0, p - 1.645 * s) for p, s in zip(predictions, stds)]
                upper_bounds = [p + 1.645 * s for p, s in zip(predictions, stds)]
                
                fig.add_trace(go.Scatter(
                    x=ages + ages[::-1],
                    y=upper_bounds + lower_bounds[::-1],
                    fill='toself',
                    fillcolor='rgba(33, 150, 243, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name="90% Confidence (GP points)",
                ))
        
        else:
            if plot_mode == "Force single architecture":
                models_to_plot = [model_type]
            else:
                models_to_plot = MODEL_TYPES
                
            colors = {"XGBoost": "#2196F3", "CatBoost": "#FF9800", "LightGBM": "#4CAF50", "Stacking": "#9C27B0", "GP": "#E64A19"}
            
            for mt in models_to_plot:
                predictions = []
                stds = []
                
                for age in ages:
                    inp = dict(filled_input)
                    inp["Age"] = float(age)
                    subset = select_subset_for_age(age)
                    
                    try:
                        model = load_model(mt, subset)
                        features_df = engineer_features(inp)
                        
                        if mt == "GP":
                            from src.gp_model import gp_predict_with_uncertainty
                            X_cols = get_feature_columns()
                            X = features_df[X_cols].values
                            mean_p, std_p = gp_predict_with_uncertainty(model, X)
                            predictions.append(float(mean_p[0]))
                            stds.append(float(std_p[0]))
                        else:
                            predictions.append(predict(model, features_df))
                            stds.append(0.0)
                    except Exception:
                        predictions.append(None)
                        stds.append(0.0)
                
                # Plot line if we have any valid predictions
                valid_idx = [i for i, p in enumerate(predictions) if p is not None]
                if valid_idx:
                    valid_ages = [ages[i] for i in valid_idx]
                    valid_preds = [predictions[i] for i in valid_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=valid_ages, y=valid_preds, mode="lines+markers",
                        name=mt,
                        line=dict(color=colors.get(mt, "#666"), width=2),
                        marker=dict(size=8),
                    ))

        # Reference lines
        for threshold in [20, 30, 40]:
            fig.add_hline(
                y=threshold, line_dash="dot", line_color="#ccc",
                annotation_text=f"{threshold} MPa", annotation_position="right",
            )

        fig.update_layout(
            xaxis_title="Age (days)",
            yaxis_title="Compressive Strength (MPa)",
            xaxis_type="log",
            xaxis=dict(
                tickmode="array", tickvals=ages,
                ticktext=[str(a) for a in ages],
            ),
            height=500, margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
        )

        st.plotly_chart(fig, width='stretch')

        # Results table
        st.subheader("Predicted Values")
        if plot_mode == "Auto-select best models":
            df_results = pd.DataFrame(results_table)
            st.dataframe(df_results, width='stretch', hide_index=True)
        else:
            st.info("Select 'Auto-select best models' to see the detailed table.")

    else:
        st.info("Enter a mix design and click Generate Curve.")
