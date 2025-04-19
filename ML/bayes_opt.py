#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例脚本：使用贝叶斯优化搜索 (col_frac, hot_frac) 以优化 SpMV 执行时间。
人工经验：
  1) 不允许 hot_frac > col_frac。
  2) 当 hot_frac == col_frac 时，仅当二者都为0或者1才合法，否则作废。
  3) 将 (0,0) 和 (1,1) 作为人工“好点” (good points) 预先测试并纳入优化初始数据。
"""
import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence


def measure_spmv_time(col_frac, hot_frac, matrix_path):
    """
    调用命令行 "../build/matxtract_perftest col_frac hot_frac matrix.mtx"
    并解析输出中的执行时间（ms）。
    返回值为执行时间(浮点数)，数值越小表示速度越快。

    人工经验：
      1) 若 hot_frac > col_frac，则返回一个大惩罚值 (1e6)。
      2) 当 hot_frac == col_frac 时，仅当二者都为0或者1才合法，否则返回大惩罚值。
    """
    if not os.path.exists(matrix_path):
        print(f"[Error] The matrix file '{matrix_path}' does not exist.")
        return 1e6
    # 1) 不允许 hot_frac > col_frac
    if hot_frac > col_frac:
        return 1e6

    # 2) 当 hot_frac == col_frac 时，仅当二者都为0或者1才合法
    if abs(hot_frac - col_frac) < 1e-15:  # 近似判断相等
        if abs(col_frac) > 1e-15 and abs(col_frac - 1.0) > 1e-15:  # 不是(0,0)或(1,1)
            return 1e6

    # ================ 如果满足上述人工经验，则正式测试 ================

    # 组装命令行参数
    cmd = [
        "../build/matxtract_perftest",
        str(col_frac),
        str(hot_frac),
        matrix_path
    ]

    try:
        # 执行命令并获取输出
        output = subprocess.check_output(cmd, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        # 如果程序执行失败，返回一个大惩罚值
        print(f"[Error] Failed to run: {cmd}\n{e}")
        return 1e6

    # 在输出中查找 "MatXtract time = xxxxx ms"
    match = re.search(r"MatXtract time\s*=\s*([\d\.Ee+-]+)\s*ms", output)
    if not match:
        # 若未找到匹配的浮点数，视为失败，返回惩罚
        print("[Warning] Could not find time in output. Full output:")
        print(output)
        return 1e6

    # 解析执行时间
    time_ms_str = match.group(1)  # 提取 "xx.xx" 部分
    try:
        time_ms = float(time_ms_str)
    except ValueError:
        # 若转换失败，也返回惩罚
        print("[Warning] Could not parse time as float.")
        return 1e6

    return time_ms

# 定义贝叶斯优化的搜索空间：col_frac, hot_frac 都在 [0,1]
# space = [
#     Real(0.0, 0.85, name='col_frac'),
#     Real(0.0, 0.7, name='hot_frac')
# ]
space = [
    Real(0.0, 1.0, name='col_frac'),
    Real(0.0, 1.0, name='hot_frac')
]


# @use_named_args(space)
# def objective(**params):
#     """
#     目标函数: 返回要最小化的值(执行时间).
#     """
#     col_frac = params['col_frac']
#     hot_frac = params['hot_frac']
#     time_ms = measure_spmv_time(col_frac, hot_frac, MATRIX_PATH)
#     return time_ms

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bayes_opt.py <matrix_path>")
        sys.exit(1)
        
    matrix_path = sys.argv[1]  # 替换原来的 MATRIX_PATH
    if not os.path.exists(matrix_path):
        print(f"[Error] The matrix file '{matrix_path}' does not exist.")
        sys.exit(1)

    print(f"Using matrix: {matrix_path}")  # 修改输出信息

    @use_named_args(space)
    def objective(**params):
        col_frac = params['col_frac']
        hot_frac = params['hot_frac']
        return measure_spmv_time(col_frac, hot_frac, matrix_path)  # 使用命令行参数
    # ========================================
    # 1) 先手动测试 (col_frac=0, hot_frac=0) 和 (col_frac=1, hot_frac=1)
    # ========================================
    init_points = [(0.0, 0.0), (1.0, 1.0)]
    x0 = []
    y0 = []
    # for col_frac, hot_frac in init_points:
    #     init_time = measure_spmv_time(col_frac, hot_frac, MATRIX_PATH)
    #     x0.append([col_frac, hot_frac])
    #     y0.append(init_time)
    #     print(f"Manually tested (col_frac={col_frac}, hot_frac={hot_frac}). Time(ms) = {init_time}")

    init0_time, init1_time = None, None  # 明确分离两个初始时间       
    # 测试 col_frac=0, hot_frac=0
    col_frac, hot_frac = init_points[0]
    time = measure_spmv_time(col_frac, hot_frac, matrix_path)
    x0.append([col_frac, hot_frac])
    y0.append(time)
    init0_time = time  # 明确赋值
    print(f"Init0 (0,0) Time = {init0_time} ms")
    # 测试 col_frac=1, hot_frac=1
    col_frac, hot_frac = init_points[1]
    time = measure_spmv_time(col_frac, hot_frac, matrix_path)
    x0.append([col_frac, hot_frac])
    y0.append(time)
    init1_time = time  # 明确赋值
    print(f"Init1 (1,1) Time = {init1_time} ms")
    # ========================================
    # 2) 进行贝叶斯优化
    #    - n_calls表示最大评估次数(可按资源酌情增减).
    #    - n_random_starts=4 => 加上 x0,y0 => 总计5个初始样本
    # ========================================
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=20,           # 总共评估 20 个点
        n_random_starts=4,    # 其中 4 个随机点 + 2个人工点 => 6个初始样本
        acq_func="EI",        # 采集函数: Expected Improvement
        random_state=42,
        x0=x0,                # 手动添加初始点
        y0=y0
    )

    # 打印优化结果
    print("===========================================")
    print("        Bayesian Optimization Result      ")
    print("===========================================")
    print(f"Best col_frac  = {res.x[0]:.4f}")
    print(f"Best hot_frac  = {res.x[1]:.4f}")
    print(f"Min Time (ms)  = {res.fun:.4f}")

    # 可选：画出收敛曲线
    # plot_convergence(res)
    # plt.title("Convergence Plot (col_frac, hot_frac) -> Time")
    # plt.savefig("convergence_plot.pdf")
    # # 关闭图形，释放内存
    # plt.close()
