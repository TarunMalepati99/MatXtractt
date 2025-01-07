#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_and_save_model.py

功能:
1) 从 ML_data_fp16.csv 读取数据并随机打乱；
2) 按 80:20 划分训练集和测试集；
3) 在训练集上做 5-Fold CV，对四个模型(RF, XGB, AdaBoost, GBDT)做性能比较；
4) 选出CV中表现最好的模型并用全部训练集拟合；
5) 在测试集上评估(MSE/MAE/R2)，并对预测结果做"阈值截断"(如<0.1视为0)，以融入"大多数矩阵(0,0)最好"的先验；
6) 最终将训练好的最佳模型保存到 "best_model.joblib"。
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
from joblib import dump  # 用于保存模型

# ====== 超参数 ======
THRESHOLD_NEAR_ZERO = 0.1  # 若预测值 < 0.1, 则视为 0

def evaluate_model(model, X, y, n_splits=5):
    """
    对给定模型进行 n_splits-Fold 交叉验证，返回 MSE 的平均值(越低越好)。
    这里对多输出回归 (col_frac, hot_frac) 做简单处理：
      - 分别计算两个目标的 MSE，然后取平均。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        # 对预测结果做先验截断(若预测值很接近0，则设为0)
        y_pred = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

        # 分别计算 col_frac, hot_frac 的 MSE
        mse_1 = mean_squared_error(y_val[:, 0], y_pred[:, 0])
        mse_2 = mean_squared_error(y_val[:, 1], y_pred[:, 1])
        # 取平均
        mse_mean = 0.5 * (mse_1 + mse_2)
        
        mse_list.append(mse_mean)
    
    return np.mean(mse_list)

def apply_zero_clamp(y_pred, threshold):
    """
    对 2D预测结果 y_pred (shape=[n_samples,2]) 进行截断:
    若 y_pred[i, j] < threshold，则 y_pred[i, j] = 0.
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
    用模型对测试集预测，并做"截断"处理，然后计算MSE/MAE/R2等指标。
    返回 (metrics_dict, y_pred_after_clamp)
    """
    y_pred = model.predict(X_test)
    # 应用截断
    y_pred_clamped = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

    # 分别计算
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
    # 1) 读取 CSV
    data_path = "ML_data_fp16.csv"  # 你的数据文件
    df = pd.read_csv(data_path)
    
    # 打乱 DataFrame 行顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2) 分离特征与目标
    feature_cols = list(df.columns)
    feature_cols.remove("MatrixName")
    feature_cols.remove("Best col_frac")
    feature_cols.remove("Best hot_frac")

    X = df[feature_cols].values
    y = df[["Best col_frac", "Best hot_frac"]].values

    # 3) 划分训练集和测试集 (80%:20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 候选模型 (不包含 MLP, SVM, LinearReg, Ridge)
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

    # 选出CV下 MSE最小的
    best_model_name = min(model_scores, key=model_scores.get)
    print(f"\n[Info] Best model from CV: {best_model_name}, MSE={model_scores[best_model_name]:.5f}")

    # 5) 用整个训练集训练该最佳模型
    best_model = candidate_models[best_model_name]
    best_model.fit(X_train, y_train)

    # 6) 在测试集上评估
    metrics, y_pred_test = final_evaluation(best_model, X_test, y_test)
    
    print("\n=== 6) Final Test Evaluation (with zero-clamp) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}")

    # 7) 保存训练好的最佳模型
    dump(best_model, "best_model.joblib")
    print("\n[Info] Model saved to best_model.joblib")


if __name__ == "__main__":
    main()
