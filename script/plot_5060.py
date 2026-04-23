import csv, json, os
import matplotlib.pyplot as plt
import numpy as np

data = {}
with open("results/benchmark_results_5060.csv") as f:
    for row in csv.DictReader(f):
        key = (row["matrix"], row["mode"], row["build"])
        data[key] = {"mean_ms": float(row["mean_ms"]),
                     "std_ms":  float(row["std_ms"]),
                     "mean_gflops": float(row["mean_gflops"])}

bayes = {}
for fname in os.listdir("results"):
    if fname.startswith("bayes_") and fname.endswith(".json"):
        with open(f"results/{fname}") as f:
            d = json.load(f)
            bayes[d["matrix"]] = d

os.makedirs("results/figures", exist_ok=True)
matrices = ["cnr-2000","web-Stanford","com-Amazon","com-Youtube","ash85","west0479","bcsstk14","poisson3Da"]

# Plot 1 - Baseline vs Optimized runtime
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, mode in zip(axes, ["Baseline", "Manual"]):
    base_t = [data.get((m, mode, "baseline"), {}).get("mean_ms", 0) for m in matrices]
    opt_t  = [data.get((m, mode, "optimized"), {}).get("mean_ms", 0) for m in matrices]
    x = np.arange(len(matrices))
    w = 0.35
    ax.bar(x-w/2, base_t, w, label="Baseline build", color="steelblue")
    ax.bar(x+w/2, opt_t,  w, label="Optimized build", color="darkorange")
    ax.set_title(f"Runtime Comparison - {mode} mode")
    ax.set_ylabel("Runtime (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/baseline_vs_optimized.png", dpi=150)
plt.close()
print("Saved: baseline_vs_optimized.png")

# Plot 2 - All 3 modes on optimized build
bayes_t = [bayes.get(m, {}).get("best_runtime_ms", 0) for m in matrices]
base_t  = [data.get((m, "Baseline", "optimized"), {}).get("mean_ms", 0) for m in matrices]
man_t   = [data.get((m, "Manual",   "optimized"), {}).get("mean_ms", 0) for m in matrices]
x = np.arange(len(matrices))
w = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x-w, base_t,  w, label="Baseline (0,0)",  color="steelblue")
ax.bar(x,   man_t,   w, label="Manual (1,1)",     color="darkorange")
ax.bar(x+w, bayes_t, w, label="BayesianBest",     color="seagreen")
ax.set_title("MatXtract RTX 5060 - Mode Comparison (Optimized Build)")
ax.set_ylabel("Runtime (ms)")
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/mode_comparison.png", dpi=150)
plt.close()
print("Saved: mode_comparison.png")

# Plot 3 - GFLOPS
base_g = [data.get((m, "Baseline", "optimized"), {}).get("mean_gflops", 0) for m in matrices]
man_g  = [data.get((m, "Manual",   "optimized"), {}).get("mean_gflops", 0) for m in matrices]
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x-w/2, base_g, w, label="Baseline (0,0)", color="steelblue")
ax.bar(x+w/2, man_g,  w, label="Manual (1,1)",   color="darkorange")
ax.set_title("GFLOPS Throughput - RTX 5060 Optimized Build")
ax.set_ylabel("GFLOPS")
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/gflops_comparison.png", dpi=150)
plt.close()
print("Saved: gflops_comparison.png")

# Plot 4 - Speedup bars
with open("results/comparison_5060.csv") as f:
    cmp = list(csv.DictReader(f))
for mode in ["Baseline", "Manual"]:
    rows = [r for r in cmp if r["mode"]==mode]
    mats = [r["matrix"] for r in rows]
    vals = [float(r["speedup"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["seagreen" if v>=1.0 else "tomato" for v in vals]
    ax.bar(mats, vals, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Speedup Optimized vs Baseline - {mode} mode")
    ax.set_ylabel("Speedup (>1 = optimized faster)")
    ax.set_xticklabels(mats, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/figures/speedup_{mode.lower()}.png", dpi=150)
    plt.close()
    print(f"Saved: speedup_{mode.lower()}.png")

# Plot 5 - BayesianBest vs Baseline runtime
bayes_mats = sorted(bayes.keys())
bayes_times = [bayes[m]["best_runtime_ms"] for m in bayes_mats]
base_times  = [data.get((m, "Baseline", "optimized"), {}).get("mean_ms", 0) for m in bayes_mats]
x = np.arange(len(bayes_mats))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x-w/2, base_times,  w, label="Baseline (0,0)", color="steelblue")
ax.bar(x+w/2, bayes_times, w, label="BayesianBest",   color="seagreen")
ax.set_title("BayesianBest vs Baseline - RTX 5060")
ax.set_ylabel("Runtime (ms)")
ax.set_xticks(x)
ax.set_xticklabels(bayes_mats, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/bayes_vs_baseline.png", dpi=150)
plt.close()
print("Saved: bayes_vs_baseline.png")

print("\nAll figures saved to results/figures/")