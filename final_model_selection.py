import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add current dir to path to allow importing src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_preprocess_data
from src.feature_engineering import engineer_features, get_feature_columns
from src.ensemble import StackingEnsemble
import catboost

def evaluate_model(model_path, is_stacking, X_test, y_test):
    print(f"Loading model from {model_path}...")
    if is_stacking:
        model = StackingEnsemble.load(model_path)
        preds = model.predict(X_test)
    else:
        # Assuming CatBoost loaded via joblib or CatBoostRegressor load_model
        if model_path.endswith('.cbm'):
            model = catboost.CatBoostRegressor()
            model.load_model(model_path)
            preds = model.predict(X_test)
        else:
            model = joblib.load(model_path)
            preds = model.predict(X_test)
            
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return rmse, r2, preds

def main():
    print("Loading data...")
    df = load_and_preprocess_data("data/Concrete_Data - Sheet1.csv")
    df_feat = engineer_features(df, verbose=False)
    feature_cols = get_feature_columns(df_feat)
    
    subsets = {
        'EA1': df_feat[df_feat['Age'] <= 3].copy(),
        'EA7': df_feat[(df_feat['Age'] > 3) & (df_feat['Age'] <= 7)].copy(),
        'EA14': df_feat[(df_feat['Age'] > 7) & (df_feat['Age'] <= 14)].copy(),
        'Full': df_feat.copy()
    }
    
    selections = {
        'EA1': {'path': 'models_stage_b/Stacking_EA1.pkl', 'is_stacking': True},
        'EA7': {'path': 'models_stage_b/Stacking_EA7.pkl', 'is_stacking': True},
        'EA14': {'path': 'models_stage_b/Stacking_EA14.pkl', 'is_stacking': True},
        'Full': {'path': 'models/CatBoost_Full.cbm', 'is_stacking': False}
    }
    
    print("=" * 50)
    print("FINAL MODEL INFERENCE CHECK")
    print("=" * 50)
    
    for subset_name, config in selections.items():
        data = subsets[subset_name]
        X = data[feature_cols].values
        y = data['Compressive_Strength'].values
        
        print(f"\nSubset: {subset_name} (n={len(data)})")
        print(f"Selected Model: {os.path.basename(config['path'])}")
        
        try:
            rmse, r2, preds = evaluate_model(config['path'], config['is_stacking'], X, y)
            print(f"  -> RMSE: {rmse:.4f}")
            print(f"  -> R2:   {r2:.4f}")
            print(f"  -> Sample Preds (first 3): {preds[:3]}")
            print(f"  -> Sample True  (first 3): {y[:3]}")
        except Exception as e:
            print(f"  -> Error evaluating: {e}")

if __name__ == '__main__':
    main()
