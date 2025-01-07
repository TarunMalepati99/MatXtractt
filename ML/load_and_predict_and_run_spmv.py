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

    # 确保 col_frac >= hot_frac，否则设置为 0
    for i in range(y_pred_clamped.shape[0]):
        if y_pred_clamped[i, 0] < y_pred_clamped[i, 1]:
            y_pred_clamped[i, 0] = 0.0
            y_pred_clamped[i, 1] = 0.0

    return y_pred_clamped

def execute_tcspmv_test(matrix_name, col_frac, hot_frac):
    """
    执行指定的命令行程序 ../build/TCSpMVlib_tcperftest。
    """
    matrix_path = f"../../../data/mtx/{matrix_name}/{matrix_name}.mtx"
    cmd = [
        "../build/TCSpMVlib_tcperftest",
        f"{col_frac}",
        f"{hot_frac}",
        matrix_path
    ]

    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        print(f"[Info] Command executed successfully for {matrix_name}")
        print(output)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed for {matrix_name}: {e}")


def main():
    # 1) 加载模型
    model = load("best_model.joblib")
    print("[Info] Loaded best_model.joblib")

    # 2) 读取 'new_data.csv'，第一列是矩阵名，后面是特征
    df_new = pd.read_csv("new_data.csv")
    matrix_names = df_new.iloc[:, 0].values  # 第一列是矩阵名
    X_new = df_new.iloc[:, 1:].values  # 其余列是特征

    # 3) 推理
    y_pred = model.predict(X_new)

    # 4) 若要体现先验(多数(0,0)最优)，则做阈值截断
    y_pred_clamped = apply_zero_clamp(y_pred, THRESHOLD_NEAR_ZERO)

    # 5) 保存或输出预测结果并执行命令
    for i, matrix_name in enumerate(matrix_names):
        col_frac = y_pred_clamped[i, 0]
        hot_frac = y_pred_clamped[i, 1]
        print(f"Matrix {matrix_name}: col_frac={col_frac:.4f}, hot_frac={hot_frac:.4f}")

        # 执行命令
        execute_tcspmv_test(matrix_name, col_frac, hot_frac)

if __name__ == "__main__":
    main()
