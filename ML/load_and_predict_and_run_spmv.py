#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import subprocess
from joblib import load

THRESHOLD_NEAR_ZERO = 0.1

def apply_zero_clamp(y_pred, threshold):
    y_pred_clamped = np.copy(y_pred)
    y_pred_clamped[y_pred_clamped < threshold] = 0.0

    # Ensure col_frac >= hot_frac, otherwise set to 0
    for i in range(y_pred_clamped.shape[0]):
        if y_pred_clamped[i, 0] < y_pred_clamped[i, 1]:
            y_pred_clamped[i, 0] = 0.0
            y_pred_clamped[i, 1] = 0.0

    return y_pred_clamped

def execute_tcspmv_test(matrix_name, col_frac, hot_frac):
    exe = r"C:\Users\tarun\MatXtract\build\matxtract_perftest.exe"
    matrix_path = rf"C:\Users\tarun\MatXtract\data\mtx\{matrix_name}\{matrix_name}.mtx"

    if col_frac == 0.0 and hot_frac == 0.0:
        cmd = [exe, matrix_path]
    else:
        cmd = [exe, str(col_frac), str(hot_frac), matrix_path]

    try:
        output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
        print(f"[Info] Command executed successfully for {matrix_name}")
        print(output)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed for {matrix_name}: {e}")
        print(e.output)
def main():
    # 1) Load model
    model = load("best_model.joblib")
    print("[Info] Loaded best_model.joblib")

    # 2) Read 'new_data.csv', first column is matrix name, rest are features
    df_new = pd.read_csv("new_data.csv")
    matrix_names = df_new.iloc[:, 0].values  # First column is matrix name
    X_new = df_new.iloc[:, 1:].values  # Remaining columns are features

    # 3) Inference
    y_pred = model.predict(X_new)

    # 4) Apply threshold clamping to reflect prior (most cases (0,0) is optimal)
    y_pred_clamped = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

    # 5) Save or output prediction results and execute commands
    for i, matrix_name in enumerate(matrix_names):
        col_frac = y_pred_clamped[i, 0]
        hot_frac = y_pred_clamped[i, 1]
        print(f"Matrix {matrix_name}: col_frac={col_frac:.4f}, hot_frac={hot_frac:.4f}")

        # Execute command
        execute_tcspmv_test(matrix_name, col_frac, hot_frac)

if __name__ == "__main__":
    main()

