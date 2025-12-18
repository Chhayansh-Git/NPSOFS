#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("experiments/final_results_table_dcor_delta03.csv")
plt.figure(figsize=(7,4))
plt.bar(df["method"], df["mean_feats"])
plt.ylabel("Average Number of Selected Features")
plt.title("Feature Count Comparison (dCor, \\delta=0.3)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("paper/figures/fig_features_by_method_dcor_delta03.png", dpi=150)
print("Saved paper/figures/fig_features_by_method_dcor_delta03.png")
