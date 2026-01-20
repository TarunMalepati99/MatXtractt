import os
import subprocess
import re
import numpy as np
import pandas as pd  # For generating CSV file
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

##############################################################################
# Modify the following path to your actual sparse matrix file root directory
##############################################################################
MATRIX_ROOT_DIR = "../data/mtx"  # Root directory containing multiple subdirectories, each with a .mtx file

OUTPUT_FILE = "batch_bayes_output.csv"  # Output CSV file path

# Initialize CSV file and write header (modified part)
# if not os.path.exists(OUTPUT_FILE):
#     df = pd.DataFrame(columns=["Matrix", "Init Time (ms)", "Best Time (ms)", "Best col_frac", "Best hot_frac"])
#     df.to_csv(OUTPUT_FILE, index
if not os.path.exists(OUTPUT_FILE):
    df = pd.DataFrame(columns=[
        "Matrix", 
        "Init0 Time (ms)",   # col_frac=0, hot_frac=0 time
        "Init1 Time (ms)",   # col_frac=1, hot_frac=1 time
        "Best Time (ms)", 
        "Best col_frac", 
        "Best hot_frac"
    ])
    df.to_csv(OUTPUT_FILE, index=False)


def measure_spmv_time(col_frac, hot_frac, matrix_path):
    """
    Call command line "../build/matxtract_perftest col_frac hot_frac matrix.mtx"
    and parse execution time (ms) from output.
    Return value is execution time (float), lower value indicates faster speed.

    Manual heuristics:
      1) If hot_frac > col_frac, return large penalty value (1e6).
      2) If hot_frac == col_frac, only valid when both are 0 or 1, otherwise return large penalty value.
    """
    if not os.path.exists(matrix_path):
        print(f"[Error] The matrix file '{matrix_path}' does not exist.")
        return 1e6

    # 1) Disallow hot_frac > col_frac
    if hot_frac > col_frac:
        return 1e6
    
    # 2) If hot_frac == col_frac, only valid when both are 0 or 1
    if abs(hot_frac - col_frac) < 1e-15:  # Approximate equality check
        if abs(col_frac) > 1e-15 and abs(col_frac - 1.0) > 1e-15:  # Not (0,0) or (1,1)
            return 1e6

    # ================ If above heuristics are satisfied, proceed with actual test ================

    # Assemble command line arguments
    cmd = [
        "../build/matxtract_perftest",
        str(col_frac),
        str(hot_frac),
        matrix_path
    ]
    
    try:
        # Execute command and get output
        output = subprocess.check_output(cmd, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        # If program execution fails, return large penalty value
        print(f"[Error] Failed to run: {cmd}\n{e}")
        return 1e6
    
    # Search for "MatXtract time = xxxxx ms" in output
    match = re.search(r"MatXtract time\s*=\s*([\d\.Ee+-]+)\s*ms", output)
    if not match:
        # If no match found for float, treat as failure and return penalty
        print("[Warning] Could not find time in output. Full output:")
        print(output)
        return 1e6
    
    # Parse execution time
    time_ms_str = match.group(1)  # Extract "xx.xx" part
    try:
        time_ms = float(time_ms_str)
    except ValueError:
        # If conversion fails, also return penalty
        print("[Warning] Could not parse time as float.")
        return 1e6
    
    return time_ms

# Define Bayesian optimization search space: col_frac, hot_frac both in [0,1]
space = [
    Real(0.0, 1.0, name='col_frac'),  # col_frac range extended to [0.0, 1.0]
    Real(0.0, 1.0, name='hot_frac')    # hot_frac range extended to [0.0, 1.0]
]

@use_named_args(space)
def objective(**params):
    """
    Objective function: return value to minimize (execution time).
    """
    col_frac = params['col_frac']
    hot_frac = params['hot_frac']
    time_ms = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
    return time_ms

if __name__ == "__main__":
    # Traverse each subdirectory under MATRIX_ROOT_DIR
    for subdir, dirs, files in os.walk(MATRIX_ROOT_DIR):
        dirs.sort()
        for dir_name in dirs:
            # if not re.match(r"^[v-z]", dir_name):
            #     continue
            matrix_file = os.path.join(subdir, dir_name, f"{dir_name}.mtx")

            # If .mtx file doesn't exist, skip
            if not os.path.exists(matrix_file):
                print(f"[Warning] Matrix file '{matrix_file}' not found. Skipping...")
                continue

            print(f"\n[INFO] Testing matrix: {matrix_file}")
            current_matrix_path = matrix_file  # Set current matrix path

            # ========================================
            # 1) First manually test (col_frac=0, hot_frac=0) and (col_frac=1, hot_frac=1)
            # ========================================
            init_points = [(0.0, 0.0), (1.0, 1.0)]
            x0 = []
            y0 = []
            # for col_frac, hot_frac in init_points:
            #     init_time = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
            #     x0.append([col_frac, hot_frac])
            #     y0.append(init_time)
            #     print(f"Manually tested (col_frac={col_frac}, hot_frac={hot_frac}). Time(ms) = {init_time}")

            init0_time, init1_time = None, None  # Explicitly separate two initial times
            
            # Test col_frac=0, hot_frac=0
            col_frac, hot_frac = init_points[0]
            time = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
            x0.append([col_frac, hot_frac])
            y0.append(time)
            init0_time = time  # Explicit assignment
            print(f"Init0 (0,0) Time = {init0_time} ms")

            # Test col_frac=1, hot_frac=1
            col_frac, hot_frac = init_points[1]
            time = measure_spmv_time(col_frac, hot_frac, current_matrix_path)
            x0.append([col_frac, hot_frac])
            y0.append(time)
            init1_time = time  # Explicit assignment
            print(f"Init1 (1,1) Time = {init1_time} ms")

            # ========================================
            # 2) Perform Bayesian optimization
            #    - n_calls indicates max evaluation count (can be adjusted based on resources).
            #    - n_random_starts=4 => plus x0,y0 => total 6 initial samples
            # ========================================
            res = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=12,           # Total 12 points to evaluate
                n_random_starts=4,    # 4 random points + 2 manual points => 6 initial samples
                acq_func="EI",        # Acquisition function: Expected Improvement
                random_state=42,
                x0=x0,                # Manually add initial points
                y0=y0
            )

            # Print optimization results
            print("===========================================")
            print("        Bayesian Optimization Result       ")
            print("===========================================")
            print(f"Best col_frac  = {res.x[0]:.4f}")
            print(f"Best hot_frac  = {res.x[1]:.4f}")
            print(f"Min Time (ms)  = {res.fun:.4f}")

            # Optional: plot convergence curve and save
            # plot_convergence(res)
            # plt.title(f"Convergence Plot: {dir_name} (col_frac, hot_frac) -> Time")
            # plt.savefig(f"{dir_name}_convergence_plot.pdf")
            # plt.close()

            # Append results to CSV file (modified part)
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
                "Init1 Time (ms)": [init1_time],  # New column
                "Best Time (ms)": [res.fun],
                "Best col_frac": [res.x[0]],
                "Best hot_frac": [res.x[1]]
            })
            df_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

    print(f"\n[INFO] All results saved to '{OUTPUT_FILE}'")
