#!/usr/bin/env python3
"""
Stage A: Early-Age Compressive Strength Prediction — ML Baseline Pipeline

Main orchestrator script. Runs the full pipeline:
  1. Data loading & preprocessing
  2. Feature engineering
  3. Early-age subset creation
  4. Baseline evaluation (Linear Regression, Random Forest)
  5. For each subset × model: Optuna HPO → Nested CV → SHAP → Plots
  6. Results summary & comparison visualizations

Usage:
    python main.py                  # Full run (100 Optuna trials, ~4-7 hours)
    python main.py --quick          # Smoke test (10 trials, ~10-20 min)
    python main.py --n-trials 50    # Custom trial count
    python main.py --resume         # Resume from last checkpoint after a crash
    python main.py --clean          # Delete checkpoints and start fresh
"""

import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import optuna

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_preprocess_data, create_age_subsets
from src.feature_engineering import engineer_features, get_feature_columns
from src.optimization import run_optimization
from src.validation import (
    nested_cv_with_optuna,
    cv_evaluation,
    group_cv_evaluation,
    evaluate_baselines,
)
from src.explainability import shap_analysis
from src.visualization import (
    plot_predictions,
    plot_residuals,
    plot_model_comparison,
    plot_performance_heatmap,
    plot_eda,
)
from src.checkpoint import TrainingCheckpoint

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Model types to evaluate
MODEL_TYPES = ['XGBoost', 'CatBoost', 'LightGBM']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage A: Early-Age Concrete Strength ML Baseline',
    )
    parser.add_argument(
        '--data', type=str, default='data/Concrete_Data - Sheet1.csv',
        help='Path to CSV dataset (default: data/Concrete_Data - Sheet1.csv)',
    )
    parser.add_argument(
        '--n-trials', type=int, default=100,
        help='Number of Optuna trials per model-subset (default: 100)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick smoke test mode: 10 trials, only Full subset',
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory for results (default: outputs)',
    )
    parser.add_argument(
        '--skip-shap', action='store_true',
        help='Skip SHAP analysis (saves ~30 min)',
    )
    parser.add_argument(
        '--skip-group-cv', action='store_true',
        help='Skip group-based (mix-level) CV evaluation',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from the last checkpoint after a crash or interruption',
    )
    parser.add_argument(
        '--clean', action='store_true',
        help='Delete existing checkpoints and start a fresh run',
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory for checkpoint files (default: checkpoints)',
    )
    return parser.parse_args()


def _create_final_model(model_type: str, best_params: dict):
    """Instantiate a model for final training (SHAP analysis)."""
    import xgboost as xgb
    from catboost import CatBoostRegressor
    import lightgbm as lgb

    if model_type == 'XGBoost':
        return xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    elif model_type == 'CatBoost':
        return CatBoostRegressor(**best_params, random_state=42, verbose=0)
    elif model_type == 'LightGBM':
        return lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()

    n_trials = 10 if args.quick else args.n_trials
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Checkpoint setup ───────────────────────────────────────────────
    ckpt = TrainingCheckpoint(
        checkpoint_dir=args.checkpoint_dir,
        pipeline_name='stage_a',
    )

    if args.clean:
        ckpt.clean()

    resumed = False
    if args.resume and ckpt.exists:
        loaded = ckpt.load()
        if loaded:
            resumed = True
            current_args = {
                'n_trials': n_trials,
                'data': args.data,
                'quick': args.quick,
            }
            ckpt.validate_args(current_args)
            print("\n  +======================================================+")
            print("  |          RESUMING FROM CHECKPOINT                   |")
            print("  +======================================================+")
            ckpt.print_status()
    else:
        # Fresh run — store pipeline args for future validation
        ckpt.pipeline_args = {
            'n_trials': n_trials,
            'data': args.data,
            'quick': args.quick,
        }

    # Use checkpoint's accumulated state if resuming, otherwise start fresh
    all_results = ckpt.all_results if resumed else []
    all_best_params = ckpt.all_best_params if resumed else {}

    # ================================================================
    print("=" * 80)
    print("  STAGE A: EARLY-AGE COMPRESSIVE STRENGTH PREDICTION — ML BASELINE")
    print("=" * 80)
    print(f"  Trials per model-subset: {n_trials}")
    print(f"  Quick mode:              {args.quick}")
    print(f"  Output directory:        {output_dir}")
    print(f"  Resume mode:             {resumed}")
    print("=" * 80)

    pipeline_start = time.time()

    # ── STEP 1: Data loading ────────────────────────────────────────────
    print("\n[1/7] Loading and preprocessing data...")
    if not os.path.exists(args.data):
        print(f"\nERROR: Dataset not found at '{args.data}'.")
        print("Please place Concrete_Data.csv in the data/ directory.")
        print("See data/README.md for expected format.")
        sys.exit(1)

    df = load_and_preprocess_data(args.data)

    # ── STEP 2: Feature engineering ─────────────────────────────────────
    print("\n[2/7] Engineering domain-derived features...")
    df_feat = engineer_features(df)
    feature_cols = get_feature_columns(df_feat)

    # ── STEP 3: EDA plots ──────────────────────────────────────────────
    print("\n[3/7] Generating EDA plots...")
    plot_eda(df, output_dir)

    # ── STEP 4: Create age subsets ──────────────────────────────────────
    print("\n[4/7] Creating early-age subsets...")
    all_subsets = create_age_subsets(df_feat)

    if args.quick:
        # In quick mode, only run on Full dataset for validation
        subsets = {'Full': all_subsets['Full']}
        print("  [QUICK MODE] Only evaluating Full subset.")
    else:
        subsets = all_subsets

    # ── STEP 5: Baseline evaluation ─────────────────────────────────────
    print("\n[5/7] Evaluating baseline models...")
    baseline_results_all = {}

    for subset_name, subset_df in subsets.items():
        baseline_key = ckpt.make_key('baseline', subset_name)

        if resumed and ckpt.is_complete(baseline_key):
            print(f"\n  [OK] Baselines for {subset_name} -- already done, skipping.")
            continue

        print(f"\n  Baselines for {subset_name} ({len(subset_df)} samples):")
        X = subset_df[feature_cols]
        y = subset_df['Compressive_Strength']

        baseline_results = evaluate_baselines(X, y)
        baseline_results_all[subset_name] = baseline_results

        baseline_rows = []
        for model_name, metrics in baseline_results.items():
            row = {
                'Subset': subset_name,
                'Model': model_name,
                'RMSE_mean': metrics['RMSE_mean'],
                'RMSE_std': metrics['RMSE_std'],
                'MAE_mean': metrics['MAE_mean'],
                'MAE_std': metrics['MAE_std'],
                'R2_mean': metrics['R2_mean'],
                'R2_std': metrics['R2_std'],
            }
            baseline_rows.append(row)

            # Prediction plots for baselines
            plot_predictions(metrics['y_true'], metrics['y_pred'],
                             model_name, subset_name, output_dir)
            plot_residuals(metrics['y_true'], metrics['y_pred'],
                           model_name, subset_name, output_dir)

        ckpt.add_results(baseline_rows)
        ckpt.mark_complete(baseline_key)
        print(f"  [OK] Checkpoint saved for baselines/{subset_name}")

    # Sync accumulated results from checkpoint
    all_results = ckpt.all_results

    # ── STEP 6: Gradient boosting models ────────────────────────────────
    print("\n[6/7] Training gradient boosting models with Optuna optimization...")

    for subset_name, subset_df in subsets.items():
        print(f"\n{'=' * 70}")
        print(f"  Subset: {subset_name} ({len(subset_df)} samples)")
        print(f"{'=' * 70}")

        X = subset_df[feature_cols]
        y = subset_df['Compressive_Strength']

        for model_type in MODEL_TYPES:
            model_key = ckpt.make_key(model_type, subset_name)

            if resumed and ckpt.is_complete(model_key):
                print(f"\n  [OK] {model_type}/{subset_name} -- already done, skipping.")
                continue

            print(f"\n  -- {model_type} --")
            model_start = time.time()

            # a) Nested cross-validation with Optuna
            print(f"  Running nested CV ({n_trials} trials × 5 outer folds)...")
            results = nested_cv_with_optuna(
                X, y, model_type,
                n_outer_folds=5, n_trials=n_trials,
                quick=args.quick
            )

            elapsed = time.time() - model_start
            print(f"\n  Nested CV Results ({elapsed:.0f}s):")
            print(f"    RMSE: {results['RMSE_mean']:.3f} +/- {results['RMSE_std']:.3f} MPa")
            print(f"    MAE:  {results['MAE_mean']:.3f} +/- {results['MAE_std']:.3f} MPa")
            print(f"    R2:   {results['R2_mean']:.4f} +/- {results['R2_std']:.4f}")
            print(f"    MAPE: {results['MAPE_mean']:.1f} +/- {results['MAPE_std']:.1f}%")
            print(f"    Max Error: {results['MaxError_mean']:.2f} +/- {results['MaxError_std']:.2f} MPa")

            # Store results via checkpoint
            result_row = {
                'Subset': subset_name,
                'Model': model_type,
                'RMSE_mean': results['RMSE_mean'],
                'RMSE_std': results['RMSE_std'],
                'MAE_mean': results['MAE_mean'],
                'MAE_std': results['MAE_std'],
                'R2_mean': results['R2_mean'],
                'R2_std': results['R2_std'],
            }

            # Use the most common best params (from fold 1 for simplicity)
            best_params = results['best_params_per_fold'][0]

            ckpt.add_results([result_row], best_params=best_params,
                             params_key=f"{model_type}_{subset_name}")

            # b) Prediction and residual plots
            plot_predictions(results['y_true'], results['y_pred'],
                             model_type, subset_name, output_dir, metrics_dict=results)
            plot_residuals(results['y_true'], results['y_pred'],
                           model_type, subset_name, output_dir)

            # c) SHAP analysis (train on last outer fold to avoid data leakage)
            if not args.skip_shap:
                print(f"  Computing SHAP analysis...")
                from sklearn.model_selection import KFold as _KFold
                _shap_cv = _KFold(n_splits=5, shuffle=True, random_state=42)
                *_, (_shap_train_idx, _shap_test_idx) = _shap_cv.split(X)
                X_shap_train = X.iloc[_shap_train_idx]
                X_shap_test = X.iloc[_shap_test_idx]
                y_shap_train = y.iloc[_shap_train_idx]
                final_model = _create_final_model(model_type, best_params)
                final_model.fit(X_shap_train, y_shap_train)
                shap_values, top_features = shap_analysis(
                    final_model, X_shap_train, X_shap_test,
                    model_type, subset_name, output_dir,
                )

            # d) Group-based CV (mix-level generalization)
            if not args.skip_group_cv and subset_name == 'Full':
                print(f"  Running group-based CV (mix-level)...")
                # Need raw df aligned with the subset
                raw_subset = df.loc[subset_df.index]
                group_results = group_cv_evaluation(
                    X, y, model_type, best_params, raw_subset,
                )
                print(f"    Group CV: RMSE={group_results['RMSE_mean']:.3f}, "
                      f"R²={group_results['R2_mean']:.4f}")

            # Mark this (model, subset) unit as complete
            ckpt.mark_complete(model_key)
            print(f"  [OK] Checkpoint saved for {model_type}/{subset_name}")

    # Sync final state from checkpoint
    all_results = ckpt.all_results
    all_best_params = ckpt.all_best_params

    # ── STEP 7: Summary & comparison ────────────────────────────────────
    print("\n[7/7] Generating final summary and comparison plots...")

    # Results DataFrame
    results_df = pd.DataFrame(all_results).round(4)

    # Separate baselines and boosting models for comparison chart
    boosting_df = results_df[
        results_df['Model'].isin(MODEL_TYPES)
    ].reset_index(drop=True)

    if len(boosting_df) > 0:
        plot_model_comparison(boosting_df, output_dir)
        plot_performance_heatmap(boosting_df, output_dir)

    # Save results CSV
    results_path = os.path.join(output_dir, 'stage_a_results_summary.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to {results_path}")

    # Save best hyperparameters
    params_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(params_path, 'w') as f:
        json.dump(all_best_params, f, indent=2)
    print(f"  Hyperparameters saved to {params_path}")

    # Print final summary table
    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 80}")
    print(f"  FINAL RESULTS SUMMARY  (Total time: {total_time / 60:.1f} min)")
    print(f"{'=' * 80}")
    print(results_df.to_string(index=False))

    # Highlight best model per subset
    print(f"\n{'-' * 50}")
    print("  Best model per subset (by R2):")
    print(f"{'-' * 50}")
    for subset in results_df['Subset'].unique():
        subset_results = results_df[results_df['Subset'] == subset]
        best_row = subset_results.loc[subset_results['R2_mean'].idxmax()]
        print(f"  {subset:<6}: {best_row['Model']:<18} "
              f"R2={best_row['R2_mean']:.4f}  "
              f"RMSE={best_row['RMSE_mean']:.3f} MPa")

    print(f"\n{'=' * 80}")
    print(f"  Pipeline complete! All outputs saved to '{output_dir}/'")
    print(f"{'=' * 80}")

    return results_df


if __name__ == '__main__':
    main()
