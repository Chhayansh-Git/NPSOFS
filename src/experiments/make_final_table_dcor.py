#!/usr/bin/env python3
"""
Create final comparison table for distance-correlation (delta=0.3) experiments.
Outputs: experiments/final_results_table_dcor_delta03.csv
"""

import pandas as pd

def main():
    # NetG-PSO results with dCor (delta=0.3)
    net = pd.read_csv("experiments/experiments_results_dcor_delta03.csv")

    # Baselines + ablations (already collected earlier)
    full = pd.read_csv("experiments/experiments_results_full.csv")

    # Combine
    df = pd.concat([net, full], ignore_index=True)

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

    out_path = "experiments/final_results_table_dcor_delta03.csv"
    summary.to_csv(out_path, index=False)

    print(summary)
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
