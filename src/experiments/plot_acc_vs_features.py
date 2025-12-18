#!/usr/bin/env python3
"""
Scatter plot of test accuracy vs selected features
Output: experiments/fig_acc_vs_features.png
"""
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("experiments/experiments_results.csv")
df2 = pd.read_csv("experiments/experiments_results_full.csv")
df = pd.concat([df1, df2], ignore_index=True)

plt.figure(figsize=(6,4))
for m in df["method"].unique():
    d = df[df["method"]==m]
    plt.scatter(d["selected_count"], d["test_acc"], label=m, alpha=0.7)

plt.xlabel("Number of Selected Features")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Feature Count")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("experiments/fig_acc_vs_features.png", dpi=150)
print("Saved experiments/fig_acc_vs_features.png")
