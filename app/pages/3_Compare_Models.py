"""
Page 3: Compare Models — Side-by-side comparison of all 12 model variants.
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles
from model_loader import (
    render_input_form, engineer_features, predict, load_model,
    load_results_df, autofill_single_input,
    MODEL_TYPES, SUBSET_NAMES,
)

apply_styles()

st.header("Compare Models")
st.markdown("Run all 12 model variants (3 models x 4 subsets) on the same input "
            "and compare predictions side by side. Check N/A for unavailable values.")

# ── Input ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6])

with left:
    st.subheader("Mix Design Input")
    raw_input = render_input_form(include_age=True, key_prefix="cmp_")
    run = st.button("Compare All Models", type="primary", use_container_width=True)

with right:
    if run:
        # Auto-fill any N/A values
        filled_input, fill_log = autofill_single_input(raw_input)
        if fill_log:
            st.info("**Auto-filled values:**\n- " + "\n- ".join(fill_log))

        features_df = engineer_features(filled_input)

        # Run all 12 predictions
        predictions = {}
        for mt in MODEL_TYPES:
            predictions[mt] = {}
            for subset in SUBSET_NAMES:
                try:
                    model = load_model(mt, subset)
                    strength = predict(model, features_df)
                    predictions[mt][subset] = round(strength, 2)
                except Exception:
                    predictions[mt][subset] = None

        # ── Table ───────────────────────────────────────────────────
        st.subheader("Predictions (MPa)")

        # Load training R2 for annotation
        results_df = load_results_df()
        boost_df = results_df[results_df["Model"].isin(MODEL_TYPES)]

        table_data = []
        for mt in MODEL_TYPES:
            row = {"Model": mt}
            for subset in SUBSET_NAMES:
                val = predictions[mt][subset]
                row[subset] = val if val is not None else "--"
            table_data.append(row)

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # ── Agreement Statistics ────────────────────────────────────
        st.subheader("Prediction Agreement")
        all_vals = [
            v for mt in predictions for v in predictions[mt].values()
            if v is not None
        ]
        if all_vals:
            arr = np.array(all_vals)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Mean", f"{arr.mean():.2f} MPa")
            with c2:
                st.metric("Std Dev", f"{arr.std():.2f} MPa")
            with c3:
                cv = (arr.std() / arr.mean() * 100) if arr.mean() != 0 else 0
                st.metric("CV", f"{cv:.1f}%")

        # ── Bar Chart ───────────────────────────────────────────────
        st.subheader("Comparison Chart")
        colors = {"XGBoost": "#2196F3", "CatBoost": "#FF9800", "LightGBM": "#4CAF50", "Stacking": "#9C27B0", "GP": "#E64A19"}

        fig = go.Figure()
        for mt in MODEL_TYPES:
            vals = [predictions[mt].get(s) for s in SUBSET_NAMES]
            fig.add_trace(go.Bar(
                name=mt,
                x=SUBSET_NAMES,
                y=vals,
                marker_color=colors.get(mt, "#666"),
            ))

        fig.update_layout(
            barmode="group",
            xaxis_title="Subset",
            yaxis_title="Predicted Strength (MPa)",
            height=400,
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Training R2 Reference ───────────────────────────────────
        st.subheader("Model Training Performance (R-squared)")
        st.caption("For context — higher R-squared indicates better training fit.")
        r2_data = []
        for mt in MODEL_TYPES:
            row = {"Model": mt}
            for subset in SUBSET_NAMES:
                match = boost_df[
                    (boost_df["Model"] == mt) & (boost_df["Subset"] == subset)
                ]
                if not match.empty:
                    row[subset] = f"{match['R2_mean'].iloc[0]:.4f}"
                else:
                    row[subset] = "--"
            r2_data.append(row)

        st.dataframe(pd.DataFrame(r2_data), use_container_width=True,
                     hide_index=True)

    else:
        st.info("Enter a mix design and click Compare All Models.")
