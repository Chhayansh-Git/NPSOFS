#!/usr/bin/env python3
"""
Boxplot of test accuracy by method
Output: experiments/fig_accuracy_by_method.png
"""
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("experiments/experiments_results.csv")
df2 = pd.read_csv("experiments/experiments_results_full.csv")
df = pd.concat([df1, df2], ignore_index=True)

plt.figure(figsize=(7,4))
df.boxplot(column="test_acc", by="method", grid=False)
plt.title("Test Accuracy Comparison by Method")
plt.suptitle("")
plt.ylabel("Test Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("experiments/fig_accuracy_by_method.png", dpi=150)
print("Saved experiments/fig_accuracy_by_method.png")
