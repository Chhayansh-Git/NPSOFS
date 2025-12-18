#!/usr/bin/env python3
"""
Create final comparison table with mean ± std for each method.
Outputs: experiments/final_results_table.csv
"""
import pandas as pd

# Load result files
base = pd.read_csv("experiments/experiments_results.csv")
full = pd.read_csv("experiments/experiments_results_full.csv")

# Combine
df = pd.concat([base, full], ignore_index=True)

summary = (
    df.groupby("method")
    .agg(
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_feats=("selected_count", "mean"),
        std_feats=("selected_count", "std"),
        mean_time=("runtime_sec", "mean")
    )
    .reset_index()
)

summary.to_csv("experiments/final_results_table.csv", index=False)
print(summary)
print("\nSaved experiments/final_results_table.csv")
