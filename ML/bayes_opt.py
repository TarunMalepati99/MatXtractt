#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script: Use Bayesian optimization to search (col_frac, hot_frac) for optimizing SpMV execution time.
Manual heuristics:
  1) Disallow hot_frac > col_frac.
  2) When hot_frac == col_frac, only valid when both are 0 or 1, otherwise invalid.
  3) Use (0,0) and (1,1) as manual "good points", pre-test and include in optimization initial data.
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
    Call command line "../build/matxtract_perftest col_frac hot_frac matrix.mtx"
    and parse execution time (ms) from output.
    Return value is execution time (float), lower value indicates faster speed.

    Manual heuristics:
      1) If hot_frac > col_frac, return large penalty value (1e6).
      2) When hot_frac == col_frac, only valid when both are 0 or 1, otherwise return large penalty value.
    """
    if not os.path.exists(matrix_path):
        print(f"[Error] The matrix file '{matrix_path}' does not exist.")
        return 1e6
    # 1) Disallow hot_frac > col_frac
    if hot_frac > col_frac:
        return 1e6

    # 2) When hot_frac == col_frac, only valid when both are 0 or 1
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
#     Objective function: return value to minimize (execution time).
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
        
    matrix_path = sys.argv[1]  # Replace original MATRIX_PATH
    if not os.path.exists(matrix_path):
        print(f"[Error] The matrix file '{matrix_path}' does not exist.")
        sys.exit(1)

    print(f"Using matrix: {matrix_path}")  # Modified output message

    @use_named_args(space)
    def objective(**params):
        col_frac = params['col_frac']
        hot_frac = params['hot_frac']
        return measure_spmv_time(col_frac, hot_frac, matrix_path)  # Use command line argument
    # ========================================
    # 1) First manually test (col_frac=0, hot_frac=0) and (col_frac=1, hot_frac=1)
    # ========================================
    init_points = [(0.0, 0.0), (1.0, 1.0)]
    x0 = []
    y0 = []
    # for col_frac, hot_frac in init_points:
    #     init_time = measure_spmv_time(col_frac, hot_frac, MATRIX_PATH)
    #     x0.append([col_frac, hot_frac])
    #     y0.append(init_time)
    #     print(f"Manually tested (col_frac={col_frac}, hot_frac={hot_frac}). Time(ms) = {init_time}")

    init0_time, init1_time = None, None  # Explicitly separate two initial times       
    # Test col_frac=0, hot_frac=0
    col_frac, hot_frac = init_points[0]
    time = measure_spmv_time(col_frac, hot_frac, matrix_path)
    x0.append([col_frac, hot_frac])
    y0.append(time)
    init0_time = time  # Explicit assignment
    print(f"Init0 (0,0) Time = {init0_time} ms")
    # Test col_frac=1, hot_frac=1
    col_frac, hot_frac = init_points[1]
    time = measure_spmv_time(col_frac, hot_frac, matrix_path)
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
        n_calls=8,           # Total 20 points to evaluate
        n_random_starts=4,    # 4 random points + 2 manual points => 6 initial samples
        acq_func="EI",        # Acquisition function: Expected Improvement
        random_state=42,
        x0=x0,                # Manually add initial points
        y0=y0
    )

    # Print optimization results
    print("===========================================")
    print("        Bayesian Optimization Result      ")
    print("===========================================")
    print(f"Best col_frac  = {res.x[0]:.4f}")
    print(f"Best hot_frac  = {res.x[1]:.4f}")
    print(f"Min Time (ms)  = {res.fun:.4f}")

    # Optional: plot convergence curve
    # plot_convergence(res)
    # plt.title("Convergence Plot (col_frac, hot_frac) -> Time")
    # plt.savefig("convergence_plot.pdf")
    # # Close figure to release memory
    # plt.close()

