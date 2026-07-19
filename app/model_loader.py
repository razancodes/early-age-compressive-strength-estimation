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
        
    if model_type == "Stacking":
        from src.ensemble import StackingEnsemble
        return StackingEnsemble.load(path)
        
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


# ═══════════════════════════════════════════════════════════════════════
# STAGE B: Best Model Auto-Selection, Domain Constants & Info Dicts
# ═══════════════════════════════════════════════════════════════════════

# Best-performing model per subset (Stage B nested CV results)
BEST_MODELS = {
    "EA1":  {"model_type": "XGBoost",  "r2": 0.8651, "rmse": 3.569,
             "display": "XGBoost (Monotonic Constrained)"},
    "EA7":  {"model_type": "GP",       "r2": 0.8623, "rmse": 4.407,
             "display": "Gaussian Process (Matérn 5/2)"},
    "EA14": {"model_type": "Stacking", "r2": 0.8708, "rmse": 4.263,
             "display": "Stacking Ensemble"},
    "Full": {"model_type": "Stacking", "r2": 0.9381, "rmse": 4.134,
             "display": "Stacking Ensemble"},
}

# Best tree-based model per subset (for SHAP when best model is GP/Stacking)
SHAP_FALLBACK = {
    "EA1":  {"model_type": "XGBoost",  "r2": 0.8651},
    "EA7":  {"model_type": "CatBoost", "r2": 0.8616},
    "EA14": {"model_type": "CatBoost", "r2": 0.8697},
    "Full": {"model_type": "CatBoost", "r2": 0.9363},
}

TREE_MODELS = ["XGBoost", "CatBoost", "LightGBM"]


def get_best_model_for_age(age: float) -> dict:
    """Return best model config for a given curing age."""
    subset = select_subset_for_age(age)
    info = BEST_MODELS[subset].copy()
    info["subset"] = subset
    return info


def compute_model_agreement(features_df: pd.DataFrame, subset: str) -> dict:
    """Run XGBoost, CatBoost, LightGBM and compute prediction agreement."""
    predictions = {}
    for mt in TREE_MODELS:
        try:
            model = load_model(mt, subset)
            predictions[mt] = predict(model, features_df)
        except Exception:
            pass
    if not predictions:
        return {"predictions": {}, "mean": None, "std": None, "cv_percent": None}
    vals = np.array(list(predictions.values()))
    mean_val = float(vals.mean())
    std_val = float(vals.std())
    cv = (std_val / mean_val * 100) if mean_val != 0 else 0.0
    return {"predictions": predictions, "mean": mean_val,
            "std": std_val, "cv_percent": cv}


# ── Concrete Grade Classification (IS 456) ─────────────────────────────

CONCRETE_GRADES = [
    (10, "M10", "Lean concrete, non-structural"),
    (15, "M15", "Plain cement concrete, footings"),
    (20, "M20", "Minimum for RCC (IS 456), residential slabs"),
    (25, "M25", "Standard structural, beams and columns"),
    (30, "M30", "Moderate exposure, commercial structures"),
    (35, "M35", "Severe exposure, bridge decks"),
    (40, "M40", "High-performance, marine structures"),
    (45, "M45", "High-performance, pre-stressed concrete"),
    (50, "M50", "Ultra-high-performance, high-rise cores"),
    (60, "M60", "Ultra-high-performance, special structures"),
    (80, "M80", "Ultra-high-performance, nuclear/defense"),
]


def classify_grade(strength_mpa: float) -> tuple:
    """Classify predicted strength into IS 456 concrete grade.
    Returns (grade_name, description, is_structural)."""
    if strength_mpa < 10:
        return ("Below M10", "Below minimum concrete grade", False)
    grade_name, grade_desc, is_structural = "M10", "Lean concrete", False
    for threshold, name, desc in CONCRETE_GRADES:
        if strength_mpa >= threshold:
            grade_name, grade_desc = name, desc
            is_structural = threshold >= 20
        else:
            break
    return (grade_name, grade_desc, is_structural)


# ── Subset / Model / Feature Info Dictionaries ─────────────────────────

SUBSET_INFO = {
    "EA1": {
        "title": "EA1 — Very Early Age (≤ 3 days)",
        "samples": 131, "age_range": "1–3 days",
        "construction_use": "Formwork removal scheduling",
        "description": (
            "Predicts compressive strength at 1 and 3 days after casting. "
            "Critical for **formwork removal decisions** — removing too early "
            "risks structural collapse. This is the hardest subset (131 samples, "
            "minimal age variation). Monotonic constraints are essential here."
        ),
    },
    "EA7": {
        "title": "EA7 — Early Age (≤ 7 days)",
        "samples": 253, "age_range": "1–7 days",
        "construction_use": "Post-tensioning and sequencing",
        "description": (
            "Predicts strength up to 7 days for **post-tensioning operations** "
            "and construction sequencing. OPC concrete typically reaches 65–75%% "
            "of 28-day strength by day 7. The Gaussian Process model excels here "
            "with native uncertainty quantification."
        ),
    },
    "EA14": {
        "title": "EA14 — Intermediate Age (≤ 14 days)",
        "samples": 315, "age_range": "1–14 days",
        "construction_use": "Quality assurance pre-check",
        "description": (
            "Predicts strength up to 14 days as an **early QA checkpoint** before "
            "the standard 28-day test. Concrete reaches ~85–90%% of 28-day strength. "
            "The Stacking Ensemble dominates here — sufficient data for the "
            "meta-learner to combine base models effectively."
        ),
    },
    "Full": {
        "title": "Full Dataset (1–365 days)",
        "samples": 1005, "age_range": "1–365 days",
        "construction_use": "General strength estimation",
        "description": (
            "Covers the **entire curing lifespan**. The Stacking Ensemble achieves "
            "R²=0.938. Note: specialized subset models outperform the Full model "
            "at early ages — the Full model overpredicts young concrete because "
            "it is dominated by 28-day samples (42%% of data)."
        ),
    },
}

MODEL_INFO = {
    "XGBoost": {
        "title": "XGBoost (Monotonic Constrained)",
        "short": "Gradient boosting with physics constraints",
        "description": (
            "eXtreme Gradient Boosting with **monotonic constraints** that force "
            "predictions to increase with cement and decrease with W/B ratio. "
            "Prevents physically impossible strength regressions. Best on EA1 "
            "(R²=0.865) where constraints regularize sparse data."
        ),
    },
    "CatBoost": {
        "title": "CatBoost (Monotonic Constrained)",
        "short": "Ordered boosting with physics constraints",
        "description": (
            "Categorical Boosting with **ordered boosting** to reduce overfitting. "
            "Monotonic constraints enforce Abrams' law. Uses symmetric trees for "
            "faster inference. Strong SHAP explainability backbone across subsets."
        ),
    },
    "LightGBM": {
        "title": "LightGBM (Monotonic Constrained)",
        "short": "Leaf-wise growth with physics constraints",
        "description": (
            "Light Gradient Boosting with **leaf-wise** tree growth for faster "
            "training. Monotonic constraints prevent non-physical predictions. "
            "5–10x faster than CatBoost with slightly lower accuracy."
        ),
    },
    "Stacking": {
        "title": "Stacking Ensemble",
        "short": "Meta-learner combining 3 GBDT models",
        "description": (
            "Two-layer architecture: XGBoost + CatBoost + LightGBM (all constrained) "
            "as base learners, with **Ridge regression** meta-learner on out-of-fold "
            "predictions. Reduces variance while preserving physics constraints. "
            "Best on EA14 (R²=0.871) and Full (R²=0.938)."
        ),
    },
    "GP": {
        "title": "Gaussian Process (Matérn 5/2)",
        "short": "Bayesian regression with native uncertainty",
        "description": (
            "Gaussian Process with **Matérn 5/2 kernel** — models smooth hydration "
            "curves with native posterior uncertainty (sigma). Provides confidence "
            "intervals without calibration. Best on EA7 (R²=0.862). Scales O(n³)."
        ),
    },
}

FEATURE_INFO = {
    "Cement": {
        "group": "Raw Input", "formula": "Direct input (kg/m³)",
        "description": "Portland cement — primary binding agent. Reacts with water to form C-S-H gel. Higher cement → higher early strength. Range: 100–550 kg/m³.",
    },
    "Blast_Furnace_Slag": {
        "group": "Raw Input", "formula": "Direct input (kg/m³)",
        "description": "GGBS — latent hydraulic SCM. Slow reaction reduces early strength but increases long-term strength and durability. Reduces heat of hydration.",
    },
    "Fly_Ash": {
        "group": "Raw Input", "formula": "Direct input (kg/m³)",
        "description": "Pozzolanic SCM requiring Ca(OH)2 to react. Negligible at ≤3 days, significant after 28 days. Improves workability and durability.",
    },
    "Water": {
        "group": "Raw Input", "formula": "Direct input (kg/m³)",
        "description": "Mixing water. Essential for hydration but excess creates capillary pores. W/B ratio is the single most important strength predictor (Abrams' law).",
    },
    "Superplasticizer": {
        "group": "Raw Input", "formula": "Direct input (kg/m³)",
        "description": "High-range water reducer. Allows lower W/B while maintaining workability. Typical dosage: 0.5–2.0%% of binder weight.",
    },
    "Coarse_Aggregate": {
        "group": "Raw Input", "formula": "Direct input (kg/m³)",
        "description": "Gravel/crushed stone (>4.75mm). Provides bulk volume and dimensional stability. Typical: 800–1150 kg/m³.",
    },
    "Fine_Aggregate": {
        "group": "Raw Input", "formula": "Direct input (kg/m³)",
        "description": "Sand (<4.75mm). Fills voids between coarse aggregate. Optimal fine-to-total ratio: 35–45%%. Typical: 590–950 kg/m³.",
    },
    "Age": {
        "group": "Raw Input", "formula": "Curing duration (days)",
        "description": "Curing age. Strength grows rapidly (1–7d), moderately (7–28d), and slowly thereafter. Standard test: 28 days (ASTM C39).",
    },
    "Binder": {
        "group": "Binder System", "formula": "Cement + Slag + Fly Ash",
        "description": "Total cementitious material. Includes OPC and all SCMs. Higher binder → higher strength but increased cost and heat.",
    },
    "W_B_ratio": {
        "group": "Binder System", "formula": "Water / Binder",
        "description": "Water-to-binder ratio — most important predictor (Abrams' law). Lower → stronger. IS 456: 0.45 (severe) to 0.55 (mild). ACI 211: 0.35–0.65.",
    },
    "GGBS_ratio": {
        "group": "Binder System", "formula": "Slag / Binder",
        "description": "Slag replacement fraction. High ratios (>50%%) slow early strength but improve 90+ day strength. IS 456 allows up to 70%% replacement.",
    },
    "FlyAsh_ratio": {
        "group": "Binder System", "formula": "Fly Ash / Binder",
        "description": "Fly ash replacement fraction. IS 456 limits to 35%%. Reduces early strength but improves long-term strength and permeability.",
    },
    "SCM_ratio": {
        "group": "Binder System", "formula": "(Slag + Fly Ash) / Binder",
        "description": "Total SCM replacement ratio. Values >0.5 mean more SCM than cement — common in green/sustainable concrete.",
    },
    "Total_Aggregate": {
        "group": "Aggregate System", "formula": "Coarse + Fine Aggregate",
        "description": "Total aggregate content. Typically 60–75%% of concrete volume. Higher aggregate → less paste, lower cost.",
    },
    "Fine_Agg_ratio": {
        "group": "Aggregate System", "formula": "Fine Aggregate / Total Aggregate",
        "description": "Fine aggregate fraction. Optimal: 0.35–0.45. Too low → harsh mix. Too high → increased water demand and shrinkage.",
    },
    "Agg_Binder_ratio": {
        "group": "Aggregate System", "formula": "Total Aggregate / Binder",
        "description": "Aggregate-to-binder ratio. Indicates mix leanness. Higher = leaner. Typical: 3.0–6.0.",
    },
    "SP_per_binder": {
        "group": "Admixture", "formula": "Superplasticizer / Binder",
        "description": "SP dosage as binder fraction. Typical: 0.005–0.025. Higher dosages enable up to 30%% water reduction.",
    },
    "log_Age": {
        "group": "Temporal", "formula": "ln(1 + Age)",
        "description": "Log-transformed age. Captures logarithmic hydration kinetics — strength is roughly linear on log-time scale after initial set.",
    },
    "sqrt_Age": {
        "group": "Temporal", "formula": "sqrt(Age)",
        "description": "Square root of age. Models intermediate growth rate between logarithmic and linear. Captures 1–28 day development.",
    },
    "Age_very_early": {
        "group": "Temporal", "formula": "1 if Age <= 3, else 0",
        "description": "Very early phase indicator. C3S reacts rapidly, producing C-S-H gel. Heat of hydration peaks. Critical for EA1 models.",
    },
    "Age_early": {
        "group": "Temporal", "formula": "1 if 3 < Age <= 7, else 0",
        "description": "Early phase indicator. C2S begins contributing. Strength reaches 65–75%% of 28-day. GGBS starts slow activation.",
    },
    "Age_standard": {
        "group": "Temporal", "formula": "1 if 7 < Age <= 28, else 0",
        "description": "Standard phase indicator. C2S increasingly active. Fly ash pozzolanic reaction begins. 28d = standard compliance age.",
    },
    "W_C_ratio": {
        "group": "Physics (Stage B)", "formula": "Water / Cement",
        "description": "Classic W/C ratio (Abrams' law). Distinct from W/B — separates OPC-dominant from SCM-dominant mixes.",
    },
    "Cement_fraction": {
        "group": "Physics (Stage B)", "formula": "Cement / Binder",
        "description": "OPC dominance indicator. 1.0 = pure OPC. Lower values = higher SCM replacement, less early strength.",
    },
    "gel_space_ratio": {
        "group": "Physics (Stage B)", "formula": "0.68*alpha*C / (0.68*alpha*C + W)",
        "description": "Powers' theory — most fundamental strength predictor. Ratio of hydrated gel to capillary space. alpha = 1-exp(-0.4*sqrt(Age)).",
    },
    "effective_WB": {
        "group": "Physics (Stage B)", "formula": "Water / (Cement + k(t)*SCMs)",
        "description": "Age-dependent effective W/B. k(t) = 0.3/0.6/0.9 for <=3/<=14/>14 days. SCMs contribute less at early ages.",
    },
    "age_wb_interaction": {
        "group": "Interaction (Stage B)", "formula": "ln(1+Age) * W/B ratio",
        "description": "Age-dependent W/B sensitivity. High W/B is more damaging at early ages when pores cannot be filled by hydration products.",
    },
    "GGBS_age_interaction": {
        "group": "Interaction (Stage B)", "formula": "GGBS_ratio * ln(1+Age)",
        "description": "Slag activation over time. GGBS barely reacts at <=3 days but contributes significantly after 28 days (latent hydraulic).",
    },
    "FlyAsh_age_interaction": {
        "group": "Interaction (Stage B)", "formula": "FlyAsh_ratio * ln(1+Age)",
        "description": "Fly ash pozzolanic delay. Even slower than slag — needs Ca(OH)2 from cement hydration. Significant after 14–28 days.",
    },
    "Binder_intensity": {
        "group": "Interaction (Stage B)", "formula": "(Cement + 0.4*Slag + 0.2*FA) / W_C_ratio",
        "description": "Weighted effective binder intensity. Weights (0.4 slag, 0.2 FA) approximate early-age reactivity. Higher = stronger, denser mix.",
    },
}

STANDARD_REFERENCES = {
    "test_age": {"value": "28 days", "source": "ASTM C39 / IS 516"},
    "wb_range_aci": {"value": "0.35 – 0.65", "source": "ACI 211"},
    "wb_max_severe": {"value": "0.45", "source": "IS 456 Table 5"},
    "wb_max_mild": {"value": "0.55", "source": "IS 456 Table 5"},
    "binder_typical": {"value": "300–450 kg/m³", "source": "IS 10262"},
    "water_typical": {"value": "160–210 kg/m³", "source": "ACI 211"},
    "min_rcc_grade": {"value": "M20 (20 MPa)", "source": "IS 456 Cl. 6.1.1"},
    "strength_7d_ratio": {"value": "65–75% of 28-day", "source": "ACI 209.2R"},
    "strength_3d_ratio": {"value": "33–40% of 28-day", "source": "ACI 209.2R"},
    "min_curing_opc": {"value": "7 days", "source": "IS 456 Cl. 13.5"},
    "min_curing_scm": {"value": "10 days", "source": "IS 456 Cl. 13.5"},
    "dataset_range": {"value": "2.33 – 82.60 MPa", "source": "UCI Dataset"},
    "dataset_mean": {"value": "35.25 +/- 16.28 MPa", "source": "UCI Dataset"},
    "dataset_samples": {"value": "1,005", "source": "UCI (25 dupes removed)"},
}


