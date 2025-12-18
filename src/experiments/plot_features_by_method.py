#!/usr/bin/env python3
"""
Bar plot of average number of selected features by method
Output: experiments/fig_features_by_method.png
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/final_results_table.csv")

plt.figure(figsize=(7,4))
plt.bar(df["method"], df["mean_feats"])
plt.ylabel("Average Number of Selected Features")
plt.title("Feature Count Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("experiments/fig_features_by_method.png", dpi=150)
print("Saved experiments/fig_features_by_method.png")
