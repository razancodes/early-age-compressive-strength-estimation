"""
Optuna hyperparameter optimization objectives for XGBoost, CatBoost, and LightGBM.

Stage B additions:
  - Monotonic constraints builder for physics-guided tree models
  - Constraint-aware objective functions
  - quick=True mode for fast smoke tests (caps trees/depth to small values)
"""

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler


# ── Monotonic Constraints (Stage B) ─────────────────────────────────────────

# Physics-based monotonicity:
#   +1 = strength increases with feature (Age, Cement)
#   -1 = strength decreases with feature (W/B ratio)
#    0 = unconstrained
MONOTONIC_MAP = {
    'Age': 1,
    'log_Age': 1,
    'sqrt_Age': 1,
    'Cement': 1,
    'W_B_ratio': -1,
    'W_C_ratio': -1,
    'effective_WB': -1,
    'gel_space_ratio': 1,
}


def build_monotonic_constraints(feature_cols: list) -> list:
    """
    Build monotonic constraint vector for the given feature columns.

    Parameters
    ----------
    feature_cols : list
        Ordered list of feature column names.

    Returns
    -------
    list
        Constraint values: +1 (increasing), -1 (decreasing), 0 (none).
    """
    return [MONOTONIC_MAP.get(col, 0) for col in feature_cols]


def build_catboost_constraints(feature_cols: list) -> dict:
    """Build CatBoost-format monotonic constraints (feature_index: direction)."""
    constraints = {}
    for i, col in enumerate(feature_cols):
        direction = MONOTONIC_MAP.get(col, 0)
        if direction != 0:
            constraints[i] = direction
    return constraints


# ── Helper: parallel CV for CatBoost ────────────────────────────────────────

def _fit_and_score(model_class, params, X_train, y_train, X_val, y_val):
    """Fit a single model fold and return RMSE."""
    model = model_class(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, y_pred)))


def _manual_cv_rmse(model_class, params, X, y, n_splits=5):
    """
    KFold CV for models incompatible with sklearn's cross_val_score.
    CatBoost 1.2.x doesn't implement __sklearn_tags__ required by sklearn >=1.8.
    Folds are run sequentially to avoid CatBoost C++ thread pool deadlocks
    when fitting concurrently in the same process. Each fit utilizes all cores.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_np = X.values if hasattr(X, 'values') else np.array(X)
    y_np = y.values if hasattr(y, 'values') else np.array(y)

    rmse_scores = []
    for train_idx, val_idx in kf.split(X_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        model = model_class(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return float(np.mean(rmse_scores))


# ── Objective Functions ─────────────────────────────────────────────────────

def objective_xgboost(trial, X, y, monotonic_constraints=None, quick=False):
    """Optuna objective for XGBoost — minimizes RMSE via 5-fold CV."""
    if quick:
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1,  # model-level threading is safe with sequential Optuna
            'tree_method': 'hist',
        }
    else:
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'n_jobs': -1,  # model-level OpenMP threading doesn't conflict with sequential Optuna
            'tree_method': 'hist',
        }

    if monotonic_constraints is not None:
        params['monotone_constraints'] = tuple(monotonic_constraints)

    model = xgb.XGBRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y, cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,  # Optuna/joblib handles outer parallelism
    )
    return float(-scores.mean())


def objective_catboost(trial, X, y, monotonic_constraints=None, quick=False):
    """Optuna objective for CatBoost — minimizes RMSE via parallel 5-fold CV."""
    if quick:
        params = {
            'depth': trial.suggest_int('depth', 2, 4),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 10, 30),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0),
            'random_state': 42,
            'verbose': 0,
        }
    else:
        params = {
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
            'random_state': 42,
            'verbose': 0,
        }

    if monotonic_constraints is not None:
        params['monotone_constraints'] = monotonic_constraints

    return _manual_cv_rmse(CatBoostRegressor, params, X, y)


def objective_lightgbm(trial, X, y, monotonic_constraints=None, quick=False):
    """Optuna objective for LightGBM — minimizes RMSE via 5-fold CV."""
    if quick:
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1,  # model-level threading is safe with sequential Optuna
            'verbose': -1,
        }
    else:
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'random_state': 42,
            'n_jobs': -1,  # model-level OpenMP threading doesn't conflict with sequential Optuna
            'verbose': -1,
        }

    if monotonic_constraints is not None:
        params['monotone_constraints'] = monotonic_constraints

    model = lgb.LGBMRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y, cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
    )
    return float(-scores.mean())


# ── Objective dispatcher ────────────────────────────────────────────────────

OBJECTIVE_MAP = {
    'XGBoost': objective_xgboost,
    'CatBoost': objective_catboost,
    'LightGBM': objective_lightgbm,
}


def run_optimization(
    X, y,
    model_type: str,
    n_trials: int = 100,
    monotonic_constraints=None,
    quick: bool = False,
) -> dict:
    """
    Run Optuna TPE optimization for the given model type.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    model_type : str
        One of 'XGBoost', 'CatBoost', 'LightGBM'.
    n_trials : int
        Number of Optuna trials.
    monotonic_constraints : list or dict, optional
        Monotonic constraints to apply during optimization.
    quick : bool
        If True, uses reduced hyperparameter search space for fast smoke tests.

    Returns
    -------
    dict
        Dictionary with 'best_params', 'best_value' (RMSE), and 'study'.
    """
    if model_type not in OBJECTIVE_MAP:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Choose from {list(OBJECTIVE_MAP.keys())}")

    objective_fn = OBJECTIVE_MAP[model_type]

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X, y, monotonic_constraints, quick=quick),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"  Best inner-CV RMSE: {study.best_value:.3f} MPa")
    print(f"  Best params: {study.best_params}")

    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study,
    }
