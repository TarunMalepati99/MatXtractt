import subprocess, csv, os, re

BUILDS = {
    "baseline":  r"build-baseline\matxtract_perftest.exe",
    "optimized": r"build-5060\matxtract_perftest.exe",
}

MODES = {
    "Baseline": (0.0, 0.0),
    "Manual":   (1.0, 1.0),
}

MATRICES = [
    r"data\mtx\cnr-2000\cnr-2000.mtx",
    r"data\mtx\web-Stanford\web-Stanford.mtx",
    r"data\mtx\com-Amazon\com-Amazon.mtx",
    r"data\mtx\com-Youtube\com-Youtube.mtx",
    r"data\mtx\ash85\ash85.mtx",
    r"data\mtx\west0479\west0479.mtx",
    r"data\mtx\bcsstk14\bcsstk14.mtx",
    r"data\mtx\poisson3Da\poisson3Da.mtx",
]

WARMUP  = 1
REPEATS = 5

def run_once(exe, col_frac, hot_frac, mtx):
    r = subprocess.run(
        [exe, str(col_frac), str(hot_frac), mtx],
        capture_output=True, text=True
    )
    t_match = re.search(r"MatXtract time\s*=\s*([\d.Ee+-]+)", r.stdout)
    g_match = re.search(r"GFLOPS\s*=\s*([\d.Ee+-]+)", r.stdout)
    t = float(t_match.group(1)) if t_match else None
    g = float(g_match.group(1)) if g_match else None
    return t, g, r.returncode

os.makedirs("results", exist_ok=True)
rows = []

for build_name, exe in BUILDS.items():
    if not os.path.exists(exe):
        print(f"MISSING EXE: {exe}"); continue
    for mtx in MATRICES:
        if not os.path.exists(mtx):
            print(f"MISSING MTX: {mtx}"); continue
        mat_name = os.path.basename(os.path.dirname(mtx))
        for mode_name, (cf, hf) in MODES.items():
            run_once(exe, cf, hf, mtx)
            times, gflops = [], []
            for _ in range(REPEATS):
                t, g, rc = run_once(exe, cf, hf, mtx)
                if rc == 0 and t is not None:
                    times.append(t)
                if g is not None:
                    gflops.append(g)
            if times:
                mean_t = sum(times) / len(times)
                std_t  = (sum((x-mean_t)**2 for x in times)/len(times))**0.5
                mean_g = sum(gflops)/len(gflops) if gflops else 0
                rows.append({
                    "build": build_name, "matrix": mat_name, "mode": mode_name,
                    "col_frac": cf, "hot_frac": hf,
                    "mean_ms": round(mean_t,4), "std_ms": round(std_t,4),
                    "mean_gflops": round(mean_g,3), "runs": len(times)
                })
                print(f"{build_name:10} | {mat_name:15} | {mode_name:10} | {mean_t:8.4f} ms +/- {std_t:.4f} | {mean_g:.2f} GFLOPS")

if rows:
    with open("results/benchmark_results_5060.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nDone. {len(rows)} rows -> results/benchmark_results_5060.csv")
else:
    print("No rows collected")
