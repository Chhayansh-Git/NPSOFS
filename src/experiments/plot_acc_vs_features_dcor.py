#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
df_net = pd.read_csv("experiments/experiments_results_dcor_delta03.csv")
df_full = pd.read_csv("experiments/experiments_results_full.csv")
df = pd.concat([df_net, df_full], ignore_index=True)
plt.figure(figsize=(6,4))
methods = df["method"].unique()
for m in methods:
    d = df[df["method"]==m]
    plt.scatter(d["selected_count"], d["test_acc"], label=m, alpha=0.7)
plt.xlabel("Number of Selected Features")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Feature Count (dCor, \\delta=0.3)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("paper/figures/fig_acc_vs_features_dcor_delta03.png", dpi=150)
print("Saved paper/figures/fig_acc_vs_features_dcor_delta03.png")
