#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_and_predict_test.py

功能:
1) 加载已保存的最佳模型 "best_model.joblib";
2) 对测试集(或其它新数据)做推理;
3) 输出预测的 (col_frac, hot_frac).
4) 同样可在预测后应用 "阈值截断" 以体现先验.
"""

import numpy as np
import pandas as pd
from joblib import load

THRESHOLD_NEAR_ZERO = 0.1  # 与训练时相同的截断阈值

# def apply_zero_clamp(y_pred, threshold):
#     y_pred_clamped = np.copy(y_pred)
#     y_pred_clamped[y_pred_clamped < threshold] = 0.0
#     return y_pred_clamped

def apply_zero_clamp(y_pred, threshold):
    y_pred_clamped = np.copy(y_pred)
    y_pred_clamped[y_pred_clamped < threshold] = 0.0

    # 确保 col_frac >= hot_frac，否则设置为 0
    for i in range(y_pred_clamped.shape[0]):
        if y_pred_clamped[i, 0] < y_pred_clamped[i, 1]:
            y_pred_clamped[i, 0] = 0.0
            y_pred_clamped[i, 1] = 0.0

    return y_pred_clamped

def main():
    # 加载训练好的模型
    model = load("best_model.joblib")
    print("[Info] Model loaded from best_model.joblib")

    # 这里演示: 依然加载 ML_data_fp16.csv 并拆分出测试集(与训练脚本保持一致)
    df = pd.read_csv("ML_data_fp16.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    feature_cols = list(df.columns)
    feature_cols.remove("MatrixName")
    feature_cols.remove("Best col_frac")
    feature_cols.remove("Best hot_frac")
    X = df[feature_cols].values
    y = df[["Best col_frac", "Best hot_frac"]].values

    # 同样的 80:20 split (确保 random_state=42 一致)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, matrix_names_train, matrix_names_test = train_test_split(
        X, y, df["MatrixName"].values, test_size=0.2, random_state=42
    )

    # 对测试集做预测
    y_pred = model.predict(X_test)
    # 做先验截断
    y_pred_clamped = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

    # 输出前10条结果做示例
    print("\n[Prediction on TestSet, first 10 samples]")
    for i in range(10):
        print(f"True=({y_test[i,0]:.3f},{y_test[i,1]:.3f}),  "
              f"Pred=({y_pred_clamped[i,0]:.3f},{y_pred_clamped[i,1]:.3f})")

    # 若需要把这些预测写到 CSV 中, 添加矩阵名称列:
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
