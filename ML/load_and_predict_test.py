#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_and_predict_test.py

Functions:
1) Load saved best model "best_model.joblib";
2) Perform inference on test set (or other new data);
3) Output predicted (col_frac, hot_frac).
4) Apply "threshold clamping" after prediction to reflect prior.
"""

import numpy as np
import pandas as pd
from joblib import load

THRESHOLD_NEAR_ZERO = 0.1  # Same threshold as training

# def apply_zero_clamp(y_pred, threshold):
#     y_pred_clamped = np.copy(y_pred)
#     y_pred_clamped[y_pred_clamped < threshold] = 0.0
#     return y_pred_clamped

def apply_zero_clamp(y_pred, threshold):
    y_pred_clamped = np.copy(y_pred)
    y_pred_clamped[y_pred_clamped < threshold] = 0.0

    # Ensure col_frac >= hot_frac, otherwise set to 0
    for i in range(y_pred_clamped.shape[0]):
        if y_pred_clamped[i, 0] < y_pred_clamped[i, 1]:
            y_pred_clamped[i, 0] = 0.0
            y_pred_clamped[i, 1] = 0.0

    return y_pred_clamped

def main():
    # Load trained model
    model = load("best_model.joblib")
    print("[Info] Model loaded from best_model.joblib")

    # Demo: Load ML_data_fp16.csv and split test set (consistent with training script)
    df = pd.read_csv("ML_data_fp16.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    feature_cols = list(df.columns)
    feature_cols.remove("MatrixName")
    feature_cols.remove("Best col_frac")
    feature_cols.remove("Best hot_frac")
    X = df[feature_cols].values
    y = df[["Best col_frac", "Best hot_frac"]].values

    # Same 80:20 split (ensure random_state=42 consistent)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, matrix_names_train, matrix_names_test = train_test_split(
        X, y, df["MatrixName"].values, test_size=0.2, random_state=42
    )

    # Make predictions on test set
    y_pred = model.predict(X_test)
    # Apply prior clamping
    y_pred_clamped = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

    # Output first 10 results as example
    print("\n[Prediction on TestSet, first 10 samples]")
    for i in range(10):
        print(f"True=({y_test[i,0]:.3f},{y_test[i,1]:.3f}),  "
              f"Pred=({y_pred_clamped[i,0]:.3f},{y_pred_clamped[i,1]:.3f})")

    # Write predictions to CSV with matrix name column if needed:
    df_test_predictions = pd.DataFrame({
        "MatrixName": matrix_names_test,
        "True col_frac": y_test[:, 0],
        "True hot_frac": y_test[:, 1],
        "Pred col_frac": y_pred_clamped[:, 0],
        "Pred hot_frac": y_pred_clamped[:, 1],
    })
    df_test_predictions.to_csv("final_test_predictions.csv", index=False)
    print("\n[Info] final_test_predictions.csv saved.")

if __name__ == "__main__":
    main()
