#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_and_save_model.py

Functions:
1) Read data from ML_data_fp64.csv and shuffle randomly;
2) Split into train/test sets with 80:20 ratio;
3) Perform 5-Fold CV on training set, compare four models (RF, XGB, AdaBoost, GBDT);
4) Select best performing model from CV and fit on entire training set;
5) Evaluate on test set (MSE/MAE/R2), apply "threshold clamping" (e.g. <0.1 as 0) to incorporate prior that most matrices perform best at (0,0);
6) Save trained best model to "best_model.joblib".
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump  # To save model

# ====== Hyperparameters ======
THRESHOLD_NEAR_ZERO = 0.1  # If prediction value < 0.1, then set to 0

def evaluate_model(model, X, y, n_splits=5):
    """
    Perform n_splits-Fold cross-validation on given model, return average MSE (lower is better).
    For multi-output regression (col_frac, hot_frac):
      - Compute MSE for each target separately, then take average.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        # Apply prior clamping to predictions (if value is close to 0, set to 0)
        y_pred = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

        # Compute MSE for col_frac and hot_frac separately
        mse_1 = mean_squared_error(y_val[:, 0], y_pred[:, 0])
        mse_2 = mean_squared_error(y_val[:, 1], y_pred[:, 1])
        # Take average
        mse_mean = 0.5 * (mse_1 + mse_2)
        
        mse_list.append(mse_mean)
    
    return np.mean(mse_list)

def apply_zero_clamp(y_pred, threshold):
    """
    Apply clamping to 2D prediction results y_pred (shape=[n_samples,2]):
    If y_pred[i, j] < threshold, then y_pred[i, j] = 0.
    """
    y_pred_clamped = np.copy(y_pred)
    y_pred_clamped[y_pred_clamped < threshold] = 0.0
    for i in range(y_pred_clamped.shape[0]):
        if y_pred_clamped[i, 0] < y_pred_clamped[i, 1]:
            y_pred_clamped[i, 0] = 0.0
            y_pred_clamped[i, 1] = 0.0
            
    return y_pred_clamped

def final_evaluation(model, X_test, y_test):
    """
    Use model to predict on test set, apply "clamping" processing, then compute MSE/MAE/R2 metrics.
    Return (metrics_dict, y_pred_after_clamp)
    """
    y_pred = model.predict(X_test)
    # Apply clamping
    y_pred_clamped = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

    # Compute separately
    mse_1 = mean_squared_error(y_test[:, 0], y_pred_clamped[:, 0])
    mse_2 = mean_squared_error(y_test[:, 1], y_pred_clamped[:, 1])
    mae_1 = mean_absolute_error(y_test[:, 0], y_pred_clamped[:, 0])
    mae_2 = mean_absolute_error(y_test[:, 1], y_pred_clamped[:, 1])
    r2_1  = r2_score(y_test[:, 0], y_pred_clamped[:, 0])
    r2_2  = r2_score(y_test[:, 1], y_pred_clamped[:, 1])

    avg_mse = 0.5*(mse_1 + mse_2)
    avg_mae = 0.5*(mae_1 + mae_2)
    avg_r2  = 0.5*(r2_1 + r2_2)

    metrics = {
        "MSE(col_frac)": mse_1,
        "MSE(hot_frac)": mse_2,
        "MAE(col_frac)": mae_1,
        "MAE(hot_frac)": mae_2,
        "R2(col_frac)":  r2_1,
        "R2(hot_frac)":  r2_2,
        "Avg_MSE":       avg_mse,
        "Avg_MAE":       avg_mae,
        "Avg_R2":        avg_r2
    }
    return metrics, y_pred_clamped


def main():
    # 1) Read CSV
    data_path = "ML_data_fp64.csv"  # Your data file
    df = pd.read_csv(data_path)
    
    # Shuffle DataFrame rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2) Separate features and targets
    feature_cols = list(df.columns)
    feature_cols.remove("MatrixName")
    feature_cols.remove("Best col_frac")
    feature_cols.remove("Best hot_frac")

    X = df[feature_cols].values
    y = df[["Best col_frac", "Best hot_frac"]].values

    # 3) Split train and test sets (80%:20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Candidate models (excluding MLP, SVM, LinearReg, Ridge)
    from sklearn.multioutput import MultiOutputRegressor
    candidate_models = {
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        "XGBoost":      MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
        "AdaBoost":     MultiOutputRegressor(AdaBoostRegressor(n_estimators=50, random_state=42)),
        "GBDT":         MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
    }

    print("=== 4) 5-Fold CV, Compare Models ===")
    model_scores = {}
    for model_name, model in candidate_models.items():
        cv_mse = evaluate_model(model, X_train, y_train, n_splits=5)
        model_scores[model_name] = cv_mse
        print(f"[CV] {model_name}, 5-Fold MSE = {cv_mse:.5f}")

    # Select model with minimum MSE from CV
    best_model_name = min(model_scores, key=model_scores.get)
    print(f"\n[Info] Best model from CV: {best_model_name}, MSE={model_scores[best_model_name]:.5f}")

    # 5) Train best model on entire training set
    best_model = candidate_models[best_model_name]
    best_model.fit(X_train, y_train)

    # 6) Evaluate on test set
    metrics, y_pred_test = final_evaluation(best_model, X_test, y_test)
    
    print("\n=== 6) Final Test Evaluation (with zero-clamp) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}")

    # 7) Save trained best model
    dump(best_model, "best_model.joblib")
    print("\n[Info] Model saved to best_model.joblib")


if __name__ == "__main__":
    main()
