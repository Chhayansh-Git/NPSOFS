#!/usr/bin/env python3
"""
Delta sweep for distance-correlation network threshold.

For each delta in `deltas`:
  - build dCor network pickle: results/colon_network_dcor_delta{:.2f}.pickle
  - count nodes/edges
  - run NetG-PSO experiments (few seeds) with that network
  - compute mean/std of selected_count and test_acc
  - write CSV: experiments/delta_sweep_results.csv
  - write LaTeX table: paper/tables/delta_sweep_table.tex

USAGE:
  python3 src/experiments/delta_sweep.py
"""
import os
import subprocess
import pickle
import time
import pandas as pd
import networkx as nx

# ---------- user parameters ----------
deltas = [0.15, 0.20, 0.25, 0.30, 0.35]
runs_per_delta = 3           # number of seeds / experiments per delta
pop_size = 12
max_iter = 20
alpha = 0.8
test_size = 0.3
data_csv = "data/colon_processed.csv"
# paths
os.makedirs("experiments", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("paper/tables", exist_ok=True)

results_rows = []

start_all = time.time()

for delta in deltas:
    tag = f"dcor_delta{int(delta*100):02d}"
    pickle_path = f"results/colon_network_{tag}.pickle"
    edgelist_csv = f"results/colon_edgelist_{tag}.csv"
    degrees_csv = f"results/colon_degrees_{tag}.csv"
    # 1) build network (calls the existing build_network.py)
    print(f"\n=== Building network for delta={delta} -> {pickle_path} ===")
    cmd_build = [
        "python3", "src/feature_network/build_network.py",
        data_csv, pickle_path,
        "--delta", str(delta)
    ]
    subprocess.run(cmd_build, check=True)
    time.sleep(0.5)
    # 2) inspect pickle to get nodes/edges (graph)
    with open(pickle_path, "rb") as f:
        meta = pickle.load(f)
    G = meta.get("graph", None)
    if G is None:
        # try keys 'graph' fallback
        raise RuntimeError(f"No graph in pickle {pickle_path}")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Network nodes={n_nodes} edges={n_edges}")
    # 3) run NetG-PSO experiments for this network
    out_csv = f"experiments/experiments_results_{tag}.csv"
    # Ensure fresh file
    if os.path.exists(out_csv):
        os.remove(out_csv)
    print(f"Running NetG-PSO for delta={delta}, writing to {out_csv}")
    cmd_run = [
        "python3", "-m", "src.experiments.experiment_driver",
        "--data", data_csv,
        "--net", pickle_path,
        "--out", out_csv,
        "--runs", str(runs_per_delta),
        "--alphas", str(alpha),
        "--pop_size", str(pop_size),
        "--max_iter", str(max_iter),
        "--test_size", str(test_size)
    ]
    t0 = time.time()
    subprocess.run(cmd_run, check=True)
    t_run = time.time() - t0
    print(f"Completed runs for delta={delta} in {t_run:.1f}s")
    # 4) read output and compute metrics
    df = pd.read_csv(out_csv)
    mean_sel = df["selected_count"].mean()
    std_sel = df["selected_count"].std()
    mean_acc = df["test_acc"].mean()
    std_acc = df["test_acc"].std()
    mean_time = df["runtime_sec"].mean()
    results_rows.append({
        "delta": delta,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_selected": float(mean_sel),
        "std_selected": float(std_sel) if not pd.isna(std_sel) else 0.0,
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc) if not pd.isna(std_acc) else 0.0,
        "mean_runtime_sec": float(mean_time)
    })

# 5) Save CSV summary
df_summary = pd.DataFrame(results_rows)
csv_out = "experiments/delta_sweep_results.csv"
df_summary.to_csv(csv_out, index=False)
print(f"\nSaved delta sweep CSV -> {csv_out}")
print(df_summary)

# 6) Write a LaTeX table file to paper/tables/delta_sweep_table.tex
tex_path = "paper/tables/delta_sweep_table.tex"
with open(tex_path, "w") as fh:
    fh.write("% Auto-generated delta sweep table (distance correlation)\n")
    fh.write("\\begin{table}[H]\n")
    fh.write("  \\centering\n")
    fh.write("  \\caption{Effect of distance-correlation threshold $\\delta$ on network density and NetG-PSO performance.}\n")
    fh.write("  \\label{tab:delta_sweep}\n")
    fh.write("  \\begin{tabular}{lrrrrr}\n")
    fh.write("    \\toprule\n")
    fh.write("    $\\delta$ & #edges & mean\\_sel & std\\_sel & mean\\_acc & std\\_acc \\\\\n")
    fh.write("    \\midrule\n")
    for r in results_rows:
        fh.write(f"    {r['delta']:.2f} & {int(r['n_edges'])} & {r['mean_selected']:.1f} & {r['std_selected']:.1f} & {r['mean_acc']:.3f} & {r['std_acc']:.3f} \\\\\n")
    fh.write("    \\bottomrule\n")
    fh.write("  \\end{tabular}\n")
    fh.write("\\end{table}\n")
print(f"Saved LaTeX table -> {tex_path}")

end_all = time.time()
print(f"All delta sweep complete in {(end_all - start_all)/60.0:.1f} minutes.")
