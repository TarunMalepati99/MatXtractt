import csv
from collections import defaultdict

data = defaultdict(dict)
with open("results/benchmark_results_5060.csv") as f:
    for row in csv.DictReader(f):
        key = (row["matrix"], row["mode"])
        data[key][row["build"]] = {
            "mean_ms": float(row["mean_ms"]),
            "std_ms":  float(row["std_ms"]),
            "mean_gflops": float(row["mean_gflops"]),
        }

# Load bayes results
bayes = {}
import json, os
for fname in os.listdir("results"):
    if fname.startswith("bayes_") and fname.endswith(".json"):
        with open(f"results/{fname}") as f:
            d = json.load(f)
            bayes[d["matrix"]] = d

print(f"{'Matrix':<15} {'Mode':<10} {'Baseline':>11} {'Optimized':>11} {'Speedup':>9} {'Status'}")
print("-" * 70)

speedups = []
rows = []
for (mat, mode), builds in sorted(data.items()):
    if "baseline" in builds and "optimized" in builds:
        b = builds["baseline"]["mean_ms"]
        o = builds["optimized"]["mean_ms"]
        s = round(b / o, 4)
        flag = "GAIN" if s > 1.0 else "REG"
        speedups.append({"matrix": mat, "mode": mode,
                         "baseline_ms": b, "optimized_ms": o,
                         "speedup": s, "status": flag})
        print(f"{mat:<15} {mode:<10} {b:>10.4f}ms {o:>10.4f}ms {s:>8.3f}x  {flag}")

vals = [x["speedup"] for x in speedups]
print(f"\nMean speedup:    {sum(vals)/len(vals):.3f}x")
print(f"Median speedup:  {sorted(vals)[len(vals)//2]:.3f}x")
print(f"Cases > 1.0:     {sum(1 for s in vals if s>1.0)}/{len(vals)}")
print(f"Best:            {max(vals):.3f}x ({next(x['matrix']+'/'+x['mode'] for x in speedups if x['speedup']==max(vals))})")
print(f"Worst:           {min(vals):.3f}x ({next(x['matrix']+'/'+x['mode'] for x in speedups if x['speedup']==min(vals))})")

print("\n--- BayesianBest Summary ---")
print(f"{'Matrix':<15} {'BestTime(ms)':>13} {'col_frac':>10} {'hot_frac':>10} {'WallTime(s)':>12}")
print("-" * 65)
for mat, d in sorted(bayes.items()):
    print(f"{mat:<15} {d['best_runtime_ms']:>13.4f} {d['best_col_frac']:>10.4f} {d['best_hot_frac']:>10.4f} {d['wall_time_s']:>12.1f}s")

with open("results/comparison_5060.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=speedups[0].keys())
    w.writeheader(); w.writerows(speedups)
print("\nSaved -> results/comparison_5060.csv")
