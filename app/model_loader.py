"""
Shared inference module.

Handles model loading, feature engineering, prediction, and SHAP computation.
All Streamlit pages import from this module.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Resolve paths relative to the project root (parent of app/)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models_stage_b")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs_stage_b")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.feature_engineering import engineer_features as _src_engineer_features

# ── Input Field Definitions ────────────────────────────────────────────
INPUT_FIELDS = [
    {"name": "Cement",            "unit": "kg/m3", "min": 100.0, "max": 550.0, "default": 280.0, "step": 5.0},
    {"name": "Blast_Furnace_Slag", "unit": "kg/m3", "min": 0.0,   "max": 360.0, "default": 50.0,  "step": 5.0},
    {"name": "Fly_Ash",           "unit": "kg/m3", "min": 0.0,   "max": 200.0, "default": 50.0,  "step": 5.0},
    {"name": "Water",             "unit": "kg/m3", "min": 120.0, "max": 250.0, "default": 180.0, "step": 5.0},
    {"name": "Superplasticizer",  "unit": "kg/m3", "min": 0.0,   "max": 32.0,  "default": 6.0,   "step": 0.5},
    {"name": "Coarse_Aggregate",  "unit": "kg/m3", "min": 800.0, "max": 1150.0,"default": 970.0, "step": 5.0},
    {"name": "Fine_Aggregate",    "unit": "kg/m3", "min": 590.0, "max": 950.0, "default": 770.0, "step": 5.0},
    {"name": "Age",               "unit": "days",  "min": 1.0,   "max": 365.0, "default": 28.0,  "step": 1.0},
]

# Display-friendly names for UI labels
DISPLAY_NAMES = {
    "Cement": "Cement",
    "Blast_Furnace_Slag": "Blast Furnace Slag",
    "Fly_Ash": "Fly Ash",
    "Water": "Water",
    "Superplasticizer": "Superplasticizer",
    "Coarse_Aggregate": "Coarse Aggregate",
    "Fine_Aggregate": "Fine Aggregate",
    "Age": "Age",
}

MODEL_TYPES = ["XGBoost", "CatBoost", "LightGBM", "Stacking", "GP"]
SUBSET_NAMES = ["EA1", "EA7", "EA14", "Full"]

# Best-practice defaults for missing values.
# Verified against actual UCI dataset statistics and ACI 211 standards.
# SCMs and SP default to 0 (not all mixes use them; 37-55% of dataset is zero).
# Age defaults to 28 days (standard test age per ASTM C39, dataset median & mode).
DEFAULT_FILL_VALUES = {
    "Cement": 281.0,              # Dataset mean 281.2, ACI 211 range 280-370
    "Blast_Furnace_Slag": 0.0,    # 45.7% of dataset is zero; optional SCM
    "Fly_Ash": 0.0,               # 55.0% of dataset is zero; optional SCM
    "Water": 182.0,               # Dataset mean 181.6, ACI 211 range 181-192
    "Superplasticizer": 0.0,      # 36.8% of dataset is zero; optional admixture
    "Coarse_Aggregate": 968.0,    # Exact dataset median, ACI 211 typical ~1024
    "Fine_Aggregate": 774.0,      # Dataset mean 773.6, ACI 211 typical ~772
    "Age": 28.0,                  # Dataset median & mode; ASTM C39 standard
}


def autofill_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Fill missing/null values with engineering best-practice defaults.

    Parameters
    ----------
    df : pd.DataFrame
        Input data that may contain NaN values.

    Returns
    -------
    tuple of (pd.DataFrame, list[str])
        Filled DataFrame and list of human-readable fill messages.
    """
    df = df.copy()
    fill_log = []

    for col, default in DEFAULT_FILL_VALUES.items():
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(default)
                fill_log.append(
                    f"{col}: {n_missing} missing values filled with "
                    f"{default} ({'0 = not used' if default == 0 else 'training median'})"
                )

    return df, fill_log


# ── Data Loading (cached) ──────────────────────────────────────────────

@st.cache_data
def load_feature_config():
    """Load feature list from feature_columns.json."""
    path = os.path.join(OUTPUTS_DIR, "feature_columns.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # Return Stage B fallback columns if file not generated yet
    return [
        "Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water", "Superplasticizer",
        "Coarse_Aggregate", "Fine_Aggregate", "Age", "Binder", "W_B_ratio",
        "GGBS_ratio", "FlyAsh_ratio", "SCM_ratio", "Total_Aggregate",
        "Fine_Agg_ratio", "Agg_Binder_ratio", "SP_per_binder", "log_Age",
        "sqrt_Age", "Age_very_early", "Age_early", "Age_standard",
        "W_C_ratio", "Cement_fraction", "gel_space_ratio", "effective_WB",
        "age_wb_interaction", "GGBS_age_interaction", "FlyAsh_age_interaction",
        "Binder_intensity"
    ]


@st.cache_data
def load_results_df():
    """Load the training results summary CSV."""
    path = os.path.join(OUTPUTS_DIR, "stage_b_results_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Return empty DataFrame with expected columns if file does not exist
    return pd.DataFrame(columns=["Subset", "Model", "RMSE_mean", "RMSE_std",
                                 "MAE_mean", "MAE_std", "R2_mean", "R2_std"])


@st.cache_data
def load_hyperparameters():
    """Load best hyperparameters JSON."""
    path = os.path.join(OUTPUTS_DIR, "best_hyperparameters_stage_b.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ── Model Loading (cached) ─────────────────────────────────────────────

@st.cache_resource
def load_model(model_type: str, subset: str):
    """
    Load a trained model from pickle file.

    Parameters
    ----------
    model_type : str
        One of 'XGBoost', 'CatBoost', 'LightGBM'.
    subset : str
        One of 'EA1', 'EA7', 'EA14', 'Full'.

    Returns
    -------
    Trained model object.
    """
    # Whitelist validation — prevent path traversal
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}. "
                         f"Must be one of {MODEL_TYPES}")
    if subset not in SUBSET_NAMES:
        raise ValueError(f"Invalid subset: {subset}. "
                         f"Must be one of {SUBSET_NAMES}")

    key = f"{model_type}_{subset}"
    path = os.path.join(MODELS_DIR, f"{key}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# ── Feature Engineering ────────────────────────────────────────────────

def engineer_features(raw_input: dict) -> pd.DataFrame:
    """
    Apply the 22 derived features (Stage B) to a raw 8-value input.

    Parameters
    ----------
    raw_input : dict
        Keys: Cement, Blast_Furnace_Slag, Fly_Ash, Water,
              Superplasticizer, Coarse_Aggregate, Fine_Aggregate, Age

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 30 features (8 raw + 22 engineered).
    """
    df = pd.DataFrame([raw_input])
    df_feat = _src_engineer_features(df, verbose=False)
    if "Compressive_Strength" in df_feat.columns:
        df_feat = df_feat.drop(columns=["Compressive_Strength"])
    return df_feat


def engineer_features_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Stage B feature engineering to a multi-row DataFrame.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame with the 8 raw columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 30 feature columns.
    """
    df_feat = _src_engineer_features(df_raw, verbose=False)
    if "Compressive_Strength" in df_feat.columns:
        df_feat = df_feat.drop(columns=["Compressive_Strength"])
    return df_feat


# ── Prediction ──────────────────────────────────────────────────────────

def get_feature_columns():
    """Return the ordered list of 30 feature columns for model input."""
    config = load_feature_config()
    if isinstance(config, dict) and "all_features" in config:
        return config["all_features"]
    return config


def predict(model, features_df: pd.DataFrame) -> float:
    """
    Run model prediction on engineered features.

    Parameters
    ----------
    model : trained model
    features_df : pd.DataFrame
        One-row DataFrame with 30 feature columns.

    Returns
    -------
    float
        Predicted compressive strength in MPa.
    """
    cols = get_feature_columns()
    X = features_df[cols]
    return float(model.predict(X)[0])


def predict_batch(model, features_df: pd.DataFrame) -> np.ndarray:
    """Predict on a multi-row DataFrame. Returns array of predictions."""
    cols = get_feature_columns()
    X = features_df[cols]
    return model.predict(X)


# ── SHAP ────────────────────────────────────────────────────────────────

def get_shap_explanation(model, features_df: pd.DataFrame):
    """
    Compute SHAP values for a single prediction.

    Returns a shap.Explanation object suitable for waterfall/force plots.
    """
    class_name = type(model).__name__
    if "Stacking" in class_name or "Pipeline" in class_name or "Ridge" in class_name:
        raise ValueError("SHAP explanations are only supported for individual tree-based models (XGBoost, CatBoost, LightGBM).")

    cols = get_feature_columns()
    X = features_df[cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X.values[0],
        feature_names=cols,
    )


# ── Subset Selection Logic ─────────────────────────────────────────────

def select_subset_for_age(age: float) -> str:
    """
    Select the most appropriate model subset for a given age.

    Uses the most specialized model available:
    - Age <= 3  -> EA1
    - Age <= 7  -> EA7
    - Age <= 14 -> EA14
    - Age > 14  -> Full
    """
    if age <= 3:
        return "EA1"
    elif age <= 7:
        return "EA7"
    elif age <= 14:
        return "EA14"
    else:
        return "Full"


# ── Plot Paths ──────────────────────────────────────────────────────────

def get_plot_path(plot_type: str, model_type: str = None, subset: str = None,
                  feature: str = None) -> str:
    """
    Build the path to a pre-computed plot image.

    Parameters
    ----------
    plot_type : str
        One of: 'pred_scatter', 'residuals', 'shap_importance',
        'shap_summary', 'shap_dep', 'eda_target_dist',
        'eda_strength_vs_age', 'eda_correlation', 'r2_heatmap',
        'model_comparison'.
    model_type : str, optional
    subset : str, optional
    feature : str, optional
        For shap_dep plots.

    Returns
    -------
    str
        Absolute path to the image file.
    """
    if plot_type in ("eda_target_dist", "eda_strength_vs_age", "eda_correlation",
                      "r2_heatmap", "model_comparison"):
        filename = f"{plot_type}.png"
    elif plot_type == "shap_dep":
        filename = f"shap_dep_{feature}_{model_type}_{subset}.png"
    else:
        filename = f"{plot_type}_{model_type}_{subset}.png"

    return os.path.join(OUTPUTS_DIR, filename)


def plot_exists(path: str) -> bool:
    """Check if a plot file exists."""
    return os.path.isfile(path)


# ── Input Form Component ───────────────────────────────────────────────

def render_input_form(include_age: bool = True, key_prefix: str = ""):
    """
    Render the 8-input form with optional N/A toggles per field.

    When "N/A" is checked, the value is set to NaN and will be
    auto-filled with the best-practice default before prediction.

    Parameters
    ----------
    include_age : bool
        Whether to include the Age field (excluded for strength curve).
    key_prefix : str
        Prefix for widget keys to avoid conflicts across pages.

    Returns
    -------
    dict
        Raw input values (float or NaN for N/A fields).
    """
    raw_input = {}

    for field in INPUT_FIELDS:
        name = field["name"]
        if name == "Age" and not include_age:
            continue

        display = DISPLAY_NAMES[name]
        label = f"{display} ({field['unit']})"

        col_input, col_na = st.columns([4, 1])

        with col_na:
            st.markdown("<br>", unsafe_allow_html=True)
            is_na = st.checkbox("N/A", key=f"{key_prefix}{name}_na",
                                help=f"Auto-fill with {DEFAULT_FILL_VALUES[name]}")

        with col_input:
            if is_na:
                st.text_input(
                    label,
                    value=f"Auto: {DEFAULT_FILL_VALUES[name]}",
                    disabled=True,
                    key=f"{key_prefix}{name}_display",
                )
                raw_input[name] = float("nan")
            else:
                val = st.number_input(
                    label,
                    min_value=field["min"],
                    max_value=field["max"],
                    value=field["default"],
                    step=field["step"],
                    key=f"{key_prefix}{name}",
                )
                raw_input[name] = val

    return raw_input


def autofill_single_input(raw_input: dict) -> tuple[dict, list[str]]:
    """
    Fill NaN values in a single-row input dict with best-practice defaults.

    Parameters
    ----------
    raw_input : dict
        Keys are field names, values are float or NaN.

    Returns
    -------
    tuple of (dict, list[str])
        Filled dict and list of human-readable messages.
    """
    filled = {}
    fill_log = []

    for name, val in raw_input.items():
        if pd.isna(val):
            default = DEFAULT_FILL_VALUES[name]
            filled[name] = default
            fill_log.append(f"{DISPLAY_NAMES[name]}: using default {default}")
        else:
            filled[name] = val

    return filled, fill_log

