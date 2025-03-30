import os
import subprocess
import re
import numpy as np
import pandas as pd  # 用于生成 CSV 文件
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

##############################################################################
# 请修改以下路径为你的实际稀疏矩阵文件根目录
##############################################################################
MATRIX_ROOT_DIR = "/home/v-wangtuowei/wangluhan/data/mtx"  # 根目录，包含多个子文件夹，每个子文件夹有一个.mtx 文件

OUTPUT_FILE = "optimization_results_fp64_dasp_.csv"  # 输出的 CSV 文件路径

# 初始化 CSV 文件，并写入表头（修改部分）
# if not os.path.exists(OUTPUT_FILE):
#     df = pd.DataFrame(columns=["Matrix", "Init Time (ms)", "Best Time (ms)", "Best col_frac", "Best hot_frac"])
#     df.to_csv(OUTPUT_FILE, index
if not os.path.exists(OUTPUT_FILE):
    df = pd.DataFrame(columns=[
        "Matrix", 
        "Init0 Time (ms)",   # col_frac=0, hot_frac=0 的时间
        "Init1 Time (ms)",   # col_frac=1, hot_frac=1 的时间
        "Best Time (ms)", 
        "Best col_frac", 
        "Best hot_frac"
    ])
    df.to_csv(OUTPUT_FILE, index=False)


def measure_spmv_time(col_frac, hot_frac, matrix_path):
    """
    调用命令行 "../build/TCSpMVlib_tcperftest col_frac hot_frac matrix.mtx"
    并解析输出中的执行时间（ms）。
    返回值为执行时间(浮点数)，数值越小表示速度越快。

    人工经验：
      1) 若 hot_frac > col_frac，则返回一个大惩罚值 (1e6)。
      2) 若 hot_frac == col_frac，则仅当二者都为0或者1时合法，否则返回大惩罚值。
    """
    if not os.path.exists(matrix_path):
        print(f"[Error] The matrix file '{matrix_path}' does not exist.")
        return 1e6

    # 1) 不允许 hot_frac > col_frac
    if hot_frac > col_frac:
        return 1e6
    
    # 2) 若 hot_frac == col_frac，则仅当二者都为0或者1时合法
    if abs(hot_frac - col_frac) < 1e-15:  # 近似判断相等
        if abs(col_frac) > 1e-15 and abs(col_frac - 1.0) > 1e-15:  # 不是(0,0)或(1,1)
            return 1e6

    # ================ 如果满足上述人工经验，则正式测试 ================

    # 组装命令行参数
    cmd = [
        "../build/TCSpMVlib_tcperftest",
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
    
    # 在输出中查找 "THE autoTC FINAL TIME = xxxxx ms"
    match = re.search(r"THE autoTC FINAL TIME\s*=\s*([\d\.Ee+-]+)\s*ms", output)
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
space = [
    Real(0.0, 1.0, name='col_frac'),  # col_frac 的范围扩展为 [0.0, 1.0]
    Real(0.0, 1.0, name='hot_frac')    # hot_frac 的范围扩展为 [0.0, 1.0]
]

@use_named_args(space)
def objective(**params):
    """
    目标函数: 返回要最小化的值(执行时间).
    """
    col_frac = params['col_frac']
    hot_frac = params['hot_frac']
    time_ms = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
    return time_ms

if __name__ == "__main__":
    # 遍历 MATRIX_ROOT_DIR 下的每个子文件夹
    for subdir, dirs, files in os.walk(MATRIX_ROOT_DIR):
        dirs.sort()
        for dir_name in dirs:
            # if not re.match(r"^[v-z]", dir_name):
            #     continue
            matrix_file = os.path.join(subdir, dir_name, f"{dir_name}.mtx")

            # 如果 .mtx 文件不存在，跳过
            if not os.path.exists(matrix_file):
                print(f"[Warning] Matrix file '{matrix_file}' not found. Skipping...")
                continue

            print(f"\n[INFO] Testing matrix: {matrix_file}")
            current_matrix_path = matrix_file  # 设置当前矩阵路径

            # ========================================
            # 1) 先手动测试 (col_frac=0, hot_frac=0) 和 (col_frac=1, hot_frac=1)
            # ========================================
            init_points = [(0.0, 0.0), (1.0, 1.0)]
            x0 = []
            y0 = []
            # for col_frac, hot_frac in init_points:
            #     init_time = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
            #     x0.append([col_frac, hot_frac])
            #     y0.append(init_time)
            #     print(f"Manually tested (col_frac={col_frac}, hot_frac={hot_frac}). Time(ms) = {init_time}")

            init0_time, init1_time = None, None  # 明确分离两个初始时间
            
            # 测试 col_frac=0, hot_frac=0
            col_frac, hot_frac = init_points[0]
            time = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
            x0.append([col_frac, hot_frac])
            y0.append(time)
            init0_time = time  # 明确赋值
            print(f"Init0 (0,0) Time = {init0_time} ms")

            # 测试 col_frac=1, hot_frac=1
            col_frac, hot_frac = init_points[1]
            time = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
            x0.append([col_frac, hot_frac])
            y0.append(time)
            init1_time = time  # 明确赋值
            print(f"Init1 (1,1) Time = {init1_time} ms")

            # ========================================
            # 2) 进行贝叶斯优化
            #    - n_calls表示最大评估次数(可按资源酌情增减).
            #    - n_random_starts=4 => 加上 x0,y0 => 总计6个初始样本
            # ========================================
            res = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=12,           # 总共评估 12 个点
                n_random_starts=4,    # 其中 4 个随机点 + 2个人工点 => 6个初始样本
                acq_func="EI",        # 采集函数: Expected Improvement
                random_state=42,
                x0=x0,                # 手动添加初始点
                y0=y0
            )

            # 打印优化结果
            print("===========================================")
            print("        Bayesian Optimization Result       ")
            print("===========================================")
            print(f"Best col_frac  = {res.x[0]:.4f}")
            print(f"Best hot_frac  = {res.x[1]:.4f}")
            print(f"Min Time (ms)  = {res.fun:.4f}")

            # 可选：画出收敛曲线并保存
            # plot_convergence(res)
            # plt.title(f"Convergence Plot: {dir_name} (col_frac, hot_frac) -> Time")
            # plt.savefig(f"{dir_name}_convergence_plot.pdf")
            # plt.close()

            # 追加结果到 CSV 文件（修改部分）
            # df_row = pd.DataFrame({
            #     "Matrix": [dir_name],
            #     "Init Time (ms)": [init_time],
            #     "Best Time (ms)": [res.fun],
            #     "Best col_frac": [res.x[0]],
            #     "Best hot_frac": [res.x[1]]
            # })
            # df_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
            df_row = pd.DataFrame({
                "Matrix": [dir_name],
                "Init0 Time (ms)": [init0_time],
                "Init1 Time (ms)": [init1_time],  # 新增列
                "Best Time (ms)": [res.fun],
                "Best col_frac": [res.x[0]],
                "Best hot_frac": [res.x[1]]
            })
            df_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

    print(f"\n[INFO] All results saved to '{OUTPUT_FILE}'")
