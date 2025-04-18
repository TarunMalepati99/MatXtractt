#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
import subprocess
import re

THRESHOLD_NEAR_ZERO = 0.1

def apply_zero_clamp(y_pred, threshold):
    y_pred_clamped = np.copy(y_pred)
    y_pred_clamped[y_pred_clamped < threshold] = 0.0

    # 确保 col_frac >= hot_frac，否则设置为 0
    for i in range(y_pred_clamped.shape[0]):
        if y_pred_clamped[i, 0] < y_pred_clamped[i, 1]:
            y_pred_clamped[i, 0] = 0.0
            y_pred_clamped[i, 1] = 0.0

    return y_pred_clamped

def evaluate_model(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_pred = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

        mse_1 = mean_squared_error(y_val[:, 0], y_pred[:, 0])
        mse_2 = mean_squared_error(y_val[:, 1], y_pred[:, 1])
        mse_mean = 0.5 * (mse_1 + mse_2)
        mse_list.append(mse_mean)

    return np.mean(mse_list)

def final_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_clamped = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

    mse_1 = mean_squared_error(y_test[:, 0], y_pred_clamped[:, 0])
    mse_2 = mean_squared_error(y_test[:, 1], y_pred_clamped[:, 1])
    mae_1 = mean_absolute_error(y_test[:, 0], y_pred_clamped[:, 0])
    mae_2 = mean_absolute_error(y_test[:, 1], y_pred_clamped[:, 1])
    r2_1  = r2_score(y_test[:, 0], y_pred_clamped[:, 0])
    r2_2  = r2_score(y_test[:, 1], y_pred_clamped[:, 1])

    metrics = {
        "MSE(col_frac)": mse_1,
        "MSE(hot_frac)": mse_2,
        "MAE(col_frac)": mae_1,
        "MAE(hot_frac)": mae_2,
        "R2(col_frac)":  r2_1,
        "R2(hot_frac)":  r2_2,
        "Avg_MSE":       0.5*(mse_1+mse_2),
        "Avg_MAE":       0.5*(mae_1+mae_2),
        "Avg_R2":        0.5*(r2_1+r2_2)
    }
    return metrics, y_pred_clamped

def execute_tcspmv_test(matrix_name, col_frac, hot_frac):
    matrix_path = f"../../../data/mtx/{matrix_name}/{matrix_name}.mtx"
    cmd = ["../build/matxtract_perftest", f"{col_frac}", f"{hot_frac}", matrix_path]

    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        match = re.search(r"FINAL TIME\s*=\s*([\d\.]+)\s*ms", output)
        if match:
            final_time_ms = float(match.group(1))
            return final_time_ms
        else:
            print(f"[Warning] FINAL TIME not found for {matrix_name}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed for {matrix_name}: {e}")
        return None

def main():
    # 加载数据
    df = pd.read_csv("ML_data_fp64.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    feature_cols = list(df.columns)
    feature_cols.remove("MatrixName")
    feature_cols.remove("Best col_frac")
    feature_cols.remove("Best hot_frac")

    X = df[feature_cols].values
    y = df[["Best col_frac", "Best hot_frac"]].values
    matrix_names = df["MatrixName"].values

    # 划分训练/测试集
    # 创建用于分层的nnz类别（如5个等级）
    df['nnz_category'] = pd.qcut(df['nnz'], q=5, labels=False)

    # 分层划分训练/测试集（stratify参数）
    X_train, X_test, y_train, y_test, matrix_train, matrix_test = train_test_split(
        X, y, matrix_names, test_size=0.2, random_state=42, stratify=df['nnz_category']
    )


    # 模型定义
    candidate_models = {
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        "XGBoost":      MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
        "AdaBoost":     MultiOutputRegressor(AdaBoostRegressor(n_estimators=50, random_state=42)),
        "GBDT":         MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
    }

    # 模型选择
    model_scores = {name: evaluate_model(model, X_train, y_train) for name, model in candidate_models.items()}
    best_model_name = min(model_scores, key=model_scores.get)
    best_model = candidate_models[best_model_name]
    print(f"Best model: {best_model_name} (MSE={model_scores[best_model_name]:.5f})")

    # 最终训练和评估
    best_model.fit(X_train, y_train)
    metrics, y_pred_test = final_evaluation(best_model, X_test, y_test)
    print("Final evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}")

    dump(best_model, "best_model.joblib")
    print("[Info] Saved best_model.joblib")

    # 测试集SpMV实际性能评测并记录
    performance_records = []
    for i, matrix_name in enumerate(matrix_test):
        col_frac, hot_frac = y_pred_test[i]
        final_time = execute_tcspmv_test(matrix_name, col_frac, hot_frac)
        if final_time is not None:
            performance_records.append({
                "MatrixName": matrix_name,
                "col_frac": col_frac,
                "hot_frac": hot_frac,
                "Final_Time_ms": final_time
            })
            print(f"[Result] {matrix_name}, Time: {final_time:.3f} ms")

    # 保存性能数据到csv
    perf_df = pd.DataFrame(performance_records)
    perf_df.to_csv("329_fp64.csv", index=False)
    print("[Info] Saved 329_fp64.csv")

if __name__ == "__main__":
    main()
