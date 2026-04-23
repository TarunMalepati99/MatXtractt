import os, subprocess, re, json, time
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

EXE = r"..\build-5060\matxtract_perftest.exe"

BAYES_CONFIG = {
    "n_calls": 12,
    "n_random_starts": 4,
    "acquisition_fn": "EI",
    "param_space": {"col_frac": [0.0, 1.0], "hot_frac": [0.0, 1.0]},
    "random_state": 42
}

MATRICES = [
    r"..\data\mtx\cnr-2000\cnr-2000.mtx",
    r"..\data\mtx\web-Stanford\web-Stanford.mtx",
    r"..\data\mtx\com-Amazon\com-Amazon.mtx",
    r"..\data\mtx\com-Youtube\com-Youtube.mtx",
    r"..\data\mtx\ash85\ash85.mtx",
    r"..\data\mtx\west0479\west0479.mtx",
    r"..\data\mtx\bcsstk14\bcsstk14.mtx",
    r"..\data\mtx\poisson3Da\poisson3Da.mtx",
]

space = [Real(0.0,1.0,name="col_frac"), Real(0.0,1.0,name="hot_frac")]

current_matrix_path = None
trial_log = []

def measure(col_frac, hot_frac, matrix_path):
    if hot_frac > col_frac: return 1e6
    if abs(hot_frac-col_frac)<1e-15 and abs(col_frac)>1e-15 and abs(col_frac-1.0)>1e-15:
        return 1e6
    try:
        out = subprocess.check_output([EXE, str(col_frac), str(hot_frac), matrix_path], universal_newlines=True)
        m = re.search(r"MatXtract time\s*=\s*([\d.Ee+-]+)", out)
        return float(m.group(1)) if m else 1e6
    except:
        return 1e6

@use_named_args(space)
def objective(**params):
    t = measure(params["col_frac"], params["hot_frac"], current_matrix_path)
    trial_log.append({"col_frac": round(params["col_frac"],4),
                      "hot_frac": round(params["hot_frac"],4), "runtime_ms": t})
    return t

os.makedirs(r"..\results", exist_ok=True)
all_results = []

for mtx in MATRICES:
    if not os.path.exists(mtx):
        print(f"MISSING: {mtx}"); continue
    mat_name = os.path.basename(os.path.dirname(mtx))
    print(f"\n[INFO] Optimizing: {mat_name}")
    current_matrix_path = mtx
    trial_log.clear()

    x0 = [[0.0,0.0],[1.0,1.0]]
    y0 = [measure(0.0,0.0,mtx), measure(1.0,1.0,mtx)]
    print(f"  Init (0,0)={y0[0]:.4f}ms  Init (1,1)={y0[1]:.4f}ms")

    t_start = time.time()
    res = gp_minimize(objective, space,
                      n_calls=BAYES_CONFIG["n_calls"],
                      n_random_starts=BAYES_CONFIG["n_random_starts"],
                      acq_func=BAYES_CONFIG["acquisition_fn"],
                      random_state=BAYES_CONFIG["random_state"],
                      x0=x0, y0=y0)
    wall_s = round(time.time()-t_start, 2)

    summary = {
        "matrix": mat_name,
        "config": BAYES_CONFIG,
        "wall_time_s": wall_s,
        "best_col_frac": round(res.x[0],4),
        "best_hot_frac": round(res.x[1],4),
        "best_runtime_ms": round(res.fun,4),
        "all_trials": trial_log[:]
    }
    with open(f"../results/bayes_{mat_name}.json","w") as f:
        json.dump(summary, f, indent=2)
    all_results.append(summary)
    print(f"  Best: col_frac={res.x[0]:.4f} hot_frac={res.x[1]:.4f} time={res.fun:.4f}ms  [{wall_s}s]")

import csv
with open(r"..\results\bayes_summary.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["matrix","best_col_frac","best_hot_frac","best_runtime_ms","wall_time_s"])
    w.writeheader()
    w.writerows([{k: r[k] for k in ["matrix","best_col_frac","best_hot_frac","best_runtime_ms","wall_time_s"]} for r in all_results])

print("\nDone. Results in results/bayes_summary.csv and results/bayes_*.json")
